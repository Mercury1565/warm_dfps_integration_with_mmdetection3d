"""
Without GPU:
python profile_step.py \
    --kitti-root ../data/kitti/2011_09_26 \
    --config ../configs/3dssd/3dssd_4x4_kitti-3d-car.py \
    --checkpoint ../configs/3dssd/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth

With GPU:
python profile_step.py \
    --kitti-root ../data/kitti/2011_09_26 \
    --config ../configs/3dssd/3dssd_4x4_kitti-3d-car.py \
    --checkpoint ../configs/3dssd/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth \
    --gpu
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import zlib
from pathlib import Path

import numpy as np
import torch
from mmdet3d.apis import inference_detector, init_model

from inference.kitti_oxts import OxtsPoseTracker
from inference.warm_dfps_manager import WarmStartManager
from inference.warm_dfps_sampler import WarmDFPSSampler


def frame_seed(drive_name: str, frame_stem: str) -> int:
    return zlib.crc32(f'{drive_name}/{frame_stem}'.encode()) & 0xffffffff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kitti-root', default='../../data/kitti/2011_09_26')
    parser.add_argument('--config', default='../../configs/3dssd/3dssd_4x4_kitti-3d-car.py')
    parser.add_argument('--checkpoint',
                        default='../../configs/3dssd/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth')
    parser.add_argument('--drive', default='2011_09_26_drive_0001_sync')
    parser.add_argument('--n-frames', type=int, default=6,
                        help='Total frames to run; frame 0 is always cold and '
                             'frame 1 is skipped as JIT/warmup noise, so '
                             'n_frames-2 frames actually get profiled.')
    parser.add_argument('--num-samples', type=int, default=4096)
    parser.add_argument('--range-adaptive', action='store_true')
    parser.add_argument('--gpu', action='store_true',
                        help='Profile step_gpu() (the Torch path) instead of '
                             'step() (the NumPy path).')
    parser.add_argument('--sort', default='cumulative',
                        help="pstats sort key, e.g. 'cumulative' or 'tottime'")
    parser.add_argument('--top', type=int, default=25)
    args = parser.parse_args()

    kitti_root = Path(args.kitti_root)
    drive_dir = kitti_root / args.drive

    model = init_model(args.config, args.checkpoint, device='cuda:0')
    sa1 = model.backbone.SA_modules[0]
    mgr = WarmStartManager(num_samples=args.num_samples or sa1.num_point[0],
                           range_adaptive=args.range_adaptive)
    warm_sampler = WarmDFPSSampler(mgr)
    sa1.points_sampler.samplers[0] = warm_sampler

    # step_gpu() is what forward() calls today; to profile step() instead,
    # monkeypatch the sampler to call it directly on the same captured P.
    if not args.gpu:
        _orig_forward = WarmDFPSSampler.forward
        def _forward_numpy_step(self, points, features, npoint):
            P = points[0].detach().cpu().numpy()
            res = self.manager.step(P, transform=self._pending_transform)
            self.last_result = res
            if res.cold:
                from mmcv.ops import furthest_point_sample
                idx = furthest_point_sample(points.contiguous(), npoint)
                S = points[0][idx[0].long()].detach().cpu().numpy()
            else:
                from inference.helpers.fps_with_preidx import farthest_point_sample_with_preidx
                preidx = torch.as_tensor(res.preidx, dtype=torch.int64, device=points.device)
                idx0 = farthest_point_sample_with_preidx(points[0], preidx, npoint)
                idx = idx0.to(torch.int32).unsqueeze(0)
                S = points[0][idx0].detach().cpu().numpy()
            self.manager.commit(S)
            self.last_S = S
            return idx
        WarmDFPSSampler.forward = _forward_numpy_step

    oxts_tracker = OxtsPoseTracker(kitti_root / 'calib_imu_to_velo.txt')
    profiler = cProfile.Profile()

    stems = [f'{i:010d}' for i in range(args.n_frames)]
    for i, stem in enumerate(stems):
        frame_path = drive_dir / 'velodyne_points' / 'data' / f'{stem}.bin'
        oxts_line = (drive_dir / 'oxts' / 'data' / f'{stem}.txt').read_text()
        transform = oxts_tracker.step(oxts_line)
        warm_sampler.set_transform(transform)
        np.random.seed(frame_seed(args.drive, stem))

        if i >= 2:  # skip frame 0 (always cold) + frame 1 (JIT/warmup noise)
            profiler.enable()
        inference_detector(model, str(frame_path))
        if i >= 2:
            profiler.disable()
        print(stem, 'cold=', warm_sampler.last_result.cold,
              'reason=', warm_sampler.last_result.reason)

    stats = pstats.Stats(profiler)
    stats.sort_stats(args.sort)
    stats.print_stats(args.top)


if __name__ == '__main__':
    main()
