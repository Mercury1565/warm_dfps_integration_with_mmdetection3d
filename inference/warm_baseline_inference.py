import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
from mmdet3d.apis import inference_detector, init_model
from kitti_oxts import OxtsPoseTracker
from warm_dfps_sampler import WarmDFPSSampler
from warm_dfps_manager import WarmStartManager


def find_drive_dirs(kitti_root: Path):
    return sorted(p for p in kitti_root.glob('*_sync') if p.is_dir())


def find_frames(drive_dir: Path):
    velo_dir = drive_dir / 'velodyne_points' / 'data'
    return sorted(velo_dir.glob('*.bin'))


def find_oxts(drive_dir: Path, frame_stem: str) -> Path:
    return drive_dir / 'oxts' / 'data' / f'{frame_stem}.txt'


def frame_seed(drive_name: str, frame_stem: str) -> int:
    """Deterministic per-frame seed for the test pipeline's PointSample
    transform -- must match batch_baseline_inference.py's frame_seed exactly,
    or baseline vs. warm runs of the same frame get different random 16384-
    point subsamples and stop being comparable regardless of D-FPS behavior.
    """
    return zlib.crc32(f'{drive_name}/{frame_stem}'.encode()) & 0xffffffff


def result_to_dict(result) -> dict:
    pred = result.pred_instances_3d
    return {
        'labels_3d': pred.labels_3d.cpu().tolist(),
        'scores_3d': pred.scores_3d.cpu().tolist(),
        'bboxes_3d': pred.bboxes_3d.tensor.cpu().tolist(),
        'box_type_3d': 'LiDAR',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kitti-root', default='data/kitti/2011_09_26',
                         help='Directory containing *_sync raw drive folders')
    parser.add_argument('--config', default='configs/3dssd/3dssd_4x4_kitti-3d-car.py')
    parser.add_argument('--checkpoint',
                         default='configs/3dssd/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth')
    parser.add_argument('--out-dir', default='warm_results')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--drives', nargs='*', default=None,
                         help='Optional subset of drive folder names to run '
                              '(e.g. 2011_09_26_drive_0001_sync). Default: all.')
    parser.add_argument('--overwrite', action='store_true',
                         help='Recompute frames even if an output JSON already exists')
    parser.add_argument('--dump-samples', action='store_true',
                         help='Also save each frame\'s SA1 sample positions and '
                              'carried-seed count to <out-dir>/<drive>/'
                              '<frame>_samples.npz (keys: S, n_seed), for BEV '
                              'comparison against the baseline run')
    parser.add_argument('--no-motion-compensation', action='store_true',
                         help='Disable oxts-based ego-motion compensation of '
                              'carried samples (reproduces the original, '
                              'uncompensated warm-D-FPS behaviour)')
    parser.add_argument('--no-range-adaptive', action='store_true',
                         help='Use one flat, global staleness/separation '
                              'threshold per frame instead of per-sample '
                              'range-adaptive thresholds (reproduces '
                              'pre-range-adaptive warm-D-FPS behaviour)')
    parser.add_argument('--imu-to-velo-calib', default=None,
                         help='Path to calib_imu_to_velo.txt. Default: '
                              '<kitti-root>/calib_imu_to_velo.txt')
    args = parser.parse_args()

    kitti_root = Path(args.kitti_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    drive_dirs = find_drive_dirs(kitti_root)
    
    if args.drives:
        drive_dirs = [d for d in drive_dirs if d.name in args.drives]
    if not drive_dirs:
        raise SystemExit(f'No drive folders found under {kitti_root}')

    print(f'Loading model: {args.config}')
    model = init_model(args.config, args.checkpoint, device=args.device)

    sa1 = model.backbone.SA_modules[0]
    mgr = WarmStartManager(
        num_samples=sa1.num_point[0], 
        range_adaptive=not args.no_range_adaptive
    )
    warm_sampler = WarmDFPSSampler(mgr)
    sa1.points_sampler.samplers[0] = warm_sampler
    print(f'Patched SA1 D-FPS -> WarmDFPSSampler (num_samples={mgr.num_samples}, '
          f'range_adaptive={mgr.range_adaptive})')

    oxts_tracker = None
    if not args.no_motion_compensation:
        calib_path = Path(args.imu_to_velo_calib or (kitti_root / 'calib_imu_to_velo.txt'))
        if not calib_path.exists():
            raise SystemExit(
                f'{calib_path} not found -- pass --imu-to-velo-calib, or rerun '
                f'with --no-motion-compensation to skip ego-motion compensation')
        oxts_tracker = OxtsPoseTracker(calib_path)
        print(f'Ego-motion compensation enabled (calib: {calib_path})')
    else:
        print('Ego-motion compensation disabled (--no-motion-compensation)')

    # Warm up CUDA/cuDNN/kernel JIT before timing; reset after so the
    # throwaway call's carried state doesn't leak into the real run.
    warmup_frames = find_frames(drive_dirs[0])
    if warmup_frames:
        t_warmup = time.time()
        inference_detector(model, str(warmup_frames[0]))
        print(f'Warm-up inference: {time.time() - t_warmup:.3f}s '
              f'(not counted in per-frame timing)')
        warm_sampler.reset()
        if oxts_tracker is not None:
            oxts_tracker.reset()

    manifest = []
    total_frames = 0
    t_start = time.time()

    for drive_dir in drive_dirs:
        frames = find_frames(drive_dir)
        if not frames:
            print(f'  [skip] {drive_dir.name}: no velodyne_points/data/*.bin found')
            continue

        drive_out = out_root / drive_dir.name
        drive_out.mkdir(parents=True, exist_ok=True)
        print(f'{drive_dir.name}: {len(frames)} frames')

        warm_sampler.reset()  # no state carries across drive boundaries
        if oxts_tracker is not None:
            oxts_tracker.reset()

        for i, frame_path in enumerate(frames):
            out_path = drive_out / f'{frame_path.stem}.json'
            if out_path.exists() and not args.overwrite:
                continue

            ego_transform = None
            if oxts_tracker is not None:
                oxts_line = find_oxts(drive_dir, frame_path.stem).read_text()
                ego_transform = oxts_tracker.step(oxts_line)
                warm_sampler.set_transform(ego_transform)

            np.random.seed(frame_seed(drive_dir.name, frame_path.stem))
            t0 = time.time()
            result, _ = inference_detector(model, str(frame_path))
            elapsed = time.time() - t0

            pred = result_to_dict(result)
            with open(out_path, 'w') as f:
                json.dump(pred, f)

            res = warm_sampler.last_result

            if args.dump_samples:
                n_seed = 0 if res.cold else int(res.preidx.size)
                np.savez(drive_out / f'{frame_path.stem}_samples.npz',
                         S=warm_sampler.last_S, n_seed=n_seed)

            manifest.append({
                'drive': drive_dir.name,
                'frame': frame_path.stem,
                'frame_index': i,
                'num_dets': len(pred['scores_3d']),
                'inference_s': elapsed,
                'cold': res.cold,
                'reason': res.reason,
                'kept_fraction': res.kept_fraction,
                'n_carried': res.n_carried,
                'n_kept': res.n_kept,
                'ego_translation_m': (float(np.linalg.norm(ego_transform[:3, 3]))
                                      if ego_transform is not None else None),
            })
            total_frames += 1

            if (i + 1) % 50 == 0:
                print(f'  {i + 1}/{len(frames)} frames done')

    with open(out_root / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - t_start
    avg = total_time / max(total_frames, 1)
    n_warm = sum(1 for m in manifest if not m['cold'])
    print(f'\nDone: {total_frames} frames across {len(drive_dirs)} drives '
          f'in {total_time:.1f}s ({avg:.3f}s/frame avg)')
    print(f'Warm frames: {n_warm}/{total_frames}')
    print(f'Predictions: {out_root}/<drive_name>/<frame_id>.json')
    print(f'Manifest (incl. cold/warm + kept_fraction per frame): {out_root / "manifest.json"}')


if __name__ == '__main__':
    main()
