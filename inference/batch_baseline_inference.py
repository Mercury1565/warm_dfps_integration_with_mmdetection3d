import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
from mmdet3d.apis import inference_detector, init_model

from logging_dfps_sampler import LoggingDFPSSampler


def find_drive_dirs(kitti_root: Path):
    return sorted(p for p in kitti_root.glob('*_sync') if p.is_dir())


def find_frames(drive_dir: Path):
    velo_dir = drive_dir / 'velodyne_points' / 'data'
    return sorted(velo_dir.glob('*.bin'))


def frame_seed(drive_name: str, frame_stem: str) -> int:
    """Deterministic per-frame seed for the test pipeline's PointSample
    transform, which downsamples the raw cloud to 16384 points via unseeded
    np.random.choice. Without pinning this, two runs of the *same* frame
    hand the network different input points and are not comparable -- keyed
    off drive+frame name (not loop position) so resumed/skipped runs still
    reseed identically. Not Python's hash(): that's randomized per-process
    unless PYTHONHASHSEED is fixed.
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
    parser.add_argument('--out-dir', default='baseline_results')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--drives', nargs='*', default=None,
                         help='Optional subset of drive folder names to run '
                              '(e.g. 2011_09_26_drive_0001_sync). Default: all.')
    parser.add_argument('--overwrite', action='store_true',
                         help='Recompute frames even if an output JSON already exists')
    parser.add_argument('--dump-samples', action='store_true',
                         help='Also save each frame\'s SA1 D-FPS sample positions '
                              'to <out-dir>/<drive>/<frame>_samples.npy, for BEV '
                              'comparison against the warm-D-FPS run')
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

    logging_sampler = None
    if args.dump_samples:
        sa1 = model.backbone.SA_modules[0]
        logging_sampler = LoggingDFPSSampler()
        sa1.points_sampler.samplers[0] = logging_sampler
        print('Patched SA1 D-FPS -> LoggingDFPSSampler (--dump-samples)')

    # Warm up CUDA context / cuDNN autotune so the first real frame's
    # timing isn't inflated by one-time init cost.
    warmup_frames = find_frames(drive_dirs[0])
    if warmup_frames:
        inference_detector(model, str(warmup_frames[0]))

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

        for i, frame_path in enumerate(frames):
            out_path = drive_out / f'{frame_path.stem}.json'
            if out_path.exists() and not args.overwrite:
                continue

            np.random.seed(frame_seed(drive_dir.name, frame_path.stem))
            t0 = time.time()
            result, _ = inference_detector(model, str(frame_path))
            elapsed = time.time() - t0

            pred = result_to_dict(result)
            with open(out_path, 'w') as f:
                json.dump(pred, f)

            if logging_sampler is not None:
                np.save(drive_out / f'{frame_path.stem}_samples.npy',
                        logging_sampler.last_S)

            manifest.append({
                'drive': drive_dir.name,
                'frame': frame_path.stem,
                'frame_index': i,
                'num_dets': len(pred['scores_3d']),
                'inference_s': elapsed,
            })
            total_frames += 1

            if (i + 1) % 50 == 0:
                print(f'  {i + 1}/{len(frames)} frames done')

    with open(out_root / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - t_start
    avg = total_time / max(total_frames, 1)
    print(f'\nDone: {total_frames} frames across {len(drive_dirs)} drives '
          f'in {total_time:.1f}s ({avg:.3f}s/frame avg)')
    print(f'Baseline predictions: {out_root}/<drive_name>/<frame_id>.json')
    print(f'Manifest: {out_root / "manifest.json"}')


if __name__ == '__main__':
    main()
