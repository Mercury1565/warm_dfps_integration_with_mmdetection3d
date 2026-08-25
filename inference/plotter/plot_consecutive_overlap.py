"""
Overlay a short window of consecutive KITTI raw velodyne sweeps to show how
little they actually move frame-to-frame -- the premise warm-D-FPS relies on
to carry SA1 samples forward instead of recomputing D-FPS from scratch.

Usage:
    # raw overlay only: each frame plotted in its own uncompensated sensor
    # frame, no transform applied -- shows the "for free" overlap that just
    # falls out of ~0.1s between sweeps
    python plot_consecutive_overlap.py \
        --kitti-root ../../data/kitti/2011_09_26 \
        --drive 2011_09_26_drive_0001_sync --frame 0000000010 \
        --num-back 2 --out-dir overlap_plots

    # also add an ego-motion-compensated panel (older frames warped into the
    # current frame's velodyne coordinates via oxts), side by side with raw
    python plot_consecutive_overlap.py \
        --kitti-root ../../data/kitti/2011_09_26 \
        --drive 2011_09_26_drive_0009_sync --frame 0000000009 \
        --num-back 2 --compensate --out-dir overlap_plots
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inference.kitti_oxts import OxtsPoseTracker


def find_frames(drive_dir: Path) -> list[Path]:
    return sorted((drive_dir / 'velodyne_points' / 'data').glob('*.bin'))


def load_points(bin_path: Path, stride: int) -> np.ndarray:
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[::stride, :3]


def frame_window(drive_dir: Path, frame: str, num_back: int) -> list[Path]:
    frames = find_frames(drive_dir)
    stems = [f.stem for f in frames]
    if frame not in stems:
        raise SystemExit(f'{frame} not found under {drive_dir / "velodyne_points" / "data"}')
    idx = stems.index(frame)
    start = max(0, idx - num_back)
    if start > idx - num_back:
        print(f'note: only {idx - start} frame(s) available before {frame}, '
              f'fewer than --num-back {num_back}')
    return frames[start:idx + 1]


def compose_to_current(drive_dir: Path, window: list[Path],
                        calib_path: Path) -> list[np.ndarray]:
    """transform[i] maps window[i]'s velodyne points into window[-1]'s
    (current frame's) velodyne coordinates. transform[-1] is identity.
    """
    tracker = OxtsPoseTracker(calib_path)
    steps = []
    for f in window:
        oxts_path = drive_dir / 'oxts' / 'data' / f'{f.stem}.txt'
        steps.append(tracker.step(oxts_path.read_text()))

    n = len(window)
    to_current = [None] * n
    acc = np.eye(4)
    for i in range(n - 1, -1, -1):
        to_current[i] = acc.copy()
        if i > 0:
            acc = acc @ steps[i]
    return to_current


def apply_transform(pts: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homo = np.concatenate([pts, np.ones((len(pts), 1), dtype=pts.dtype)], axis=1)
    return (homo @ transform.T)[:, :3]


def nn_min_dist(a: np.ndarray, b: np.ndarray, chunk: int = 500) -> np.ndarray:
    """For each point in `a` (xy only), its distance to the nearest point in
    `b`. Chunked over `a` so memory stays bounded regardless of len(b).
    """
    out = np.empty(len(a), dtype=np.float32)
    for i in range(0, len(a), chunk):
        d = a[i:i + chunk, None, :2] - b[None, :, :2]
        out[i:i + chunk] = np.sqrt((d ** 2).sum(-1)).min(axis=1)
    return out


def subsample(pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(pts) <= n:
        return pts
    idx = rng.choice(len(pts), size=n, replace=False)
    return pts[idx]


def plot_panel(ax, labeled_pts: list[tuple[str, np.ndarray]], title: str) -> None:
    n = len(labeled_pts)
    for k, (label, pts) in enumerate(labeled_pts[:-1]):
        shade = 0.25 + 0.55 * (k / max(n - 2, 1))
        ax.scatter(pts[:, 0], pts[:, 1], s=0.3, linewidths=0,
                   c=[plt.cm.Blues(shade)], label=label, zorder=k + 1)
    cur_label, cur_pts = labeled_pts[-1]
    ax.scatter(cur_pts[:, 0], cur_pts[:, 1], s=0.3, c='#d62728', linewidths=0,
               label=cur_label, zorder=n)
    ax.set_xlabel('x (m, forward)')
    ax.set_ylabel('y (m, left)')
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8, markerscale=15)


def report_overlap(window: list[Path], pts_by_frame: list[np.ndarray],
                    threshold: float, nn_sample: int, tag: str) -> None:
    rng = np.random.default_rng(0)
    cur = subsample(pts_by_frame[-1], nn_sample, rng)
    print(f'\n[{tag}] fraction of current-frame points ({len(cur)} sampled) '
          f'within {threshold:.2f} m of a point in each older frame:')
    for k in range(len(window) - 1):
        older = subsample(pts_by_frame[k], nn_sample, rng)
        d = nn_min_dist(cur, older)
        frac = float((d <= threshold).mean())
        offset = len(window) - 1 - k
        print(f'  t-{offset} ({window[k].stem}): {frac:.1%} within threshold, '
              f'median nearest-neighbor dist = {np.median(d):.3f} m')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kitti-root', required=True)
    ap.add_argument('--drive', required=True)
    ap.add_argument('--frame', required=True, help='Current (most recent) frame stem, e.g. 0000000010')
    ap.add_argument('--num-back', type=int, default=2, help='Number of preceding frames to overlay')
    ap.add_argument('--point-stride', type=int, default=4,
                     help='Subsample point clouds for plotting speed')
    ap.add_argument('--compensate', action='store_true',
                     help='Also plot an ego-motion-compensated panel (older frames warped '
                          "into the current frame's velodyne coordinates via oxts)")
    ap.add_argument('--imu-to-velo-calib', default=None,
                     help='Path to calib_imu_to_velo.txt. Default: <kitti-root>/calib_imu_to_velo.txt')
    ap.add_argument('--nn-threshold', type=float, default=0.3,
                     help='Distance (m) under which a point counts as "closely positioned" '
                          'to some point in the other frame')
    ap.add_argument('--nn-sample', type=int, default=3000,
                     help='Points sampled per frame for the nearest-neighbor overlap metric')
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--xlim', type=float, nargs=2, default=None)
    ap.add_argument('--ylim', type=float, nargs=2, default=None)
    args = ap.parse_args()

    kitti_root = Path(args.kitti_root)
    drive_dir = kitti_root / args.drive
    window = frame_window(drive_dir, args.frame, args.num_back)

    raw_pts = [load_points(f, args.point_stride) for f in window]
    labels = [f't-{len(window) - 1 - k}' if k < len(window) - 1 else 't (current)'
              for k in range(len(window))]

    n_cols = 2 if args.compensate else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(10 * n_cols, 10), squeeze=False)
    axes = axes[0]

    plot_panel(axes[0], list(zip(labels, raw_pts)),
               f'{args.drive} / {args.frame} -- raw overlay, no ego-motion compensation')
    report_overlap(window, raw_pts, args.nn_threshold, args.nn_sample, tag='raw')

    if args.compensate:
        calib_path = Path(args.imu_to_velo_calib or (kitti_root / 'calib_imu_to_velo.txt'))
        if not calib_path.exists():
            raise SystemExit(f'{calib_path} not found -- pass --imu-to-velo-calib')
        transforms = compose_to_current(drive_dir, window, calib_path)
        comp_pts = [apply_transform(p, t) for p, t in zip(raw_pts, transforms)]

        plot_panel(axes[1], list(zip(labels, comp_pts)),
                   f'{args.drive} / {args.frame} -- ego-motion compensated')
        report_overlap(window, comp_pts, args.nn_threshold, args.nn_sample, tag='compensated')

    for ax in axes:
        if args.xlim:
            ax.set_xlim(args.xlim)
        if args.ylim:
            ax.set_ylim(args.ylim)

    fig.tight_layout()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'overlap_{args.drive}_{args.frame}_back{args.num_back}.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
