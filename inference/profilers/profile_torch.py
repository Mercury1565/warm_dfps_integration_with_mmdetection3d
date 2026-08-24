"""torch.profiler-based profiling: unlike profile_step.py (cProfile, Python
call-time only), this sees the actual CUDA timeline -- kernel launches,
Memcpy HtoD/DtoH, and CPU<->GPU sync points. Use this specifically to spot
host/device round-trips, not just "which Python function is slow."

Usage:
    python profile_torch.py --gpu --n-frames 6
    python profile_torch.py --n-frames 6              # profile the old NumPy step() path instead
"""
from __future__ import annotations

import argparse
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from mmdet3d.apis import inference_detector, init_model
from torch.profiler import ProfilerActivity, profile, record_function

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
                        help='Total frames to run and profile. A separate '
                             'throwaway warm-up call runs before profiling '
                             'starts, so one-time CUDA/cuDNN/JIT cost does '
                             'not pollute the profiled frames, including '
                             'frame 0.')
    parser.add_argument('--num-samples', type=int, default=4096)
    parser.add_argument('--range-adaptive', action='store_true')
    parser.add_argument('--gpu', action='store_true',
                        help='Profile step_gpu() instead of the old NumPy step().')

    args = parser.parse_args()

    kitti_root = Path(args.kitti_root)
    drive_dir = kitti_root / args.drive

    model = init_model(args.config, args.checkpoint, device='cuda:0')
    sa1 = model.backbone.SA_modules[0]
    mgr = WarmStartManager(num_samples=args.num_samples or sa1.num_point[0],
                           range_adaptive=args.range_adaptive)
    warm_sampler = WarmDFPSSampler(mgr)
    sa1.points_sampler.samplers[0] = warm_sampler

    if not args.gpu:
        def _forward_numpy_step(self, points, features, npoint):
            with record_function("manager.step (numpy)"):
                P = points[0].detach().cpu().numpy()
                res = self.manager.step(P, transform=self._pending_transform)
            self.last_result = res

            if res.cold:
                with record_function("cold: mmcv furthest_point_sample"):
                    from mmcv.ops import furthest_point_sample
                    idx = furthest_point_sample(points.contiguous(), npoint)
                    S = points[0][idx[0].long()].detach().cpu().numpy()
            else:
                with record_function("warm: fps_refill CUDA kernel"):
                    from inference.helpers.fps_with_preidx import farthest_point_sample_with_preidx
                    preidx = torch.as_tensor(res.preidx, dtype=torch.int64, device=points.device)
                    idx0 = farthest_point_sample_with_preidx(points[0], preidx, npoint)
                    idx = idx0.to(torch.int32).unsqueeze(0)
                    S = points[0][idx0].detach().cpu().numpy()
                    
            with record_function("manager.commit"):
                self.manager.commit(S)
            self.last_S = S
            return idx
        WarmDFPSSampler.forward = _forward_numpy_step

    oxts_tracker = OxtsPoseTracker(kitti_root / 'calib_imu_to_velo.txt')
    stems = [f'{i:010d}' for i in range(args.n_frames)]

    # Warm up CUDA/cuDNN/kernel JIT before profiling; reset after so this
    # throwaway call's carried state doesn't leak into the profiled run.
    warmup_path = drive_dir / 'velodyne_points' / 'data' / f'{stems[0]}.bin'
    t_warmup = time.time()
    inference_detector(model, str(warmup_path))
    print(f'Warm-up inference: {time.time() - t_warmup:.3f}s (not profiled)')
    warm_sampler.reset()
    oxts_tracker.reset()

    def run_frame(stem: str):
        frame_path = drive_dir / 'velodyne_points' / 'data' / f'{stem}.bin'
        oxts_line = (drive_dir / 'oxts' / 'data' / f'{stem}.txt').read_text()
        transform = oxts_tracker.step(oxts_line)
        warm_sampler.set_transform(transform)
        np.random.seed(frame_seed(args.drive, stem))
        with record_function(f"frame_{stem}"):
            inference_detector(model, str(frame_path))
        print(stem, 'cold=', warm_sampler.last_result.cold,
              'reason=', warm_sampler.last_result.reason)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False) as prof:
        for stem in stems:
            run_frame(stem)

    print(prof.key_averages().table(sort_by="cuda_time_total"))
    print()
    print("--- sorted by self CPU time (spots Python/host-side overhead, incl. syncs) ---")
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

if __name__ == '__main__':
    main()
