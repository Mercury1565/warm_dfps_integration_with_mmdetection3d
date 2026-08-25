"""
Usage:
    # explicit frame(s) on one drive
    python plot_bev.py \
        --kitti-root data/kitti/2011_09_26 \
        --baseline-dir bev_baseline_results --warm-dir bev_warm_results \
        --drive 2011_09_26_drive_0001_sync --frame 0000000042 0000000043

    # or auto-pick N worst-recall frames from the highest-retention (Q4)
    # bucket, using a compare_results.py --dump per-frame json -- across all
    # drives present in bev_baseline_results/bev_warm_results by default, or
    # restricted to one drive if --drive is given
    python plot_bev.py \
        --kitti-root data/kitti/2011_09_26 \
        --baseline-dir bev_baseline_results --warm-dir bev_warm_results \
        --num-frames 5 --per-frame-dump per_frame.json

    # or every common frame between baseline and warm (optionally restricted
    # to one drive with --drive) -- generates one PNG per frame, so this can
    # be slow/disk-heavy over a full multi-thousand-frame drive
    python plot_bev.py \
        --kitti-root ../../data/kitti/2011_09_26 \
        --baseline-dir ../results/baseline_results \
        --warm-dir ../results_6/warm_results \
        --drive 2011_09_26_drive_0001_sync --frame 0000000040 0000000041 0000000042 0000000043 0000000044 \
        --out-dir bev_plots_all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inference.comparison_scripts.comparison_result import load_manifest


def load_points(bin_path: Path, stride: int) -> np.ndarray:
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[::stride, :2]


def load_boxes(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = json.load(open(json_path))
    boxes = np.array(d['bboxes_3d'], dtype=np.float32).reshape(-1, 7)
    scores = np.array(d['scores_3d'], dtype=np.float32)
    return boxes, scores


def box_bev_corners(box: np.ndarray) -> np.ndarray:
    """box = [x, y, z, dx, dy, dz, yaw] -> (4, 2) BEV corner ring."""
    x, y, _, dx, dy, _, yaw = box
    local = np.array([[dx / 2, dy / 2], [dx / 2, -dy / 2],
                       [-dx / 2, -dy / 2], [-dx / 2, dy / 2]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([x, y])


def draw_boxes(ax, boxes: np.ndarray, color: str, label: str,
               scores: np.ndarray | None = None, linestyle: str = '-',
               zorder: int = 2) -> None:
    for i, box in enumerate(boxes):
        ring = box_bev_corners(box)
        ring = np.vstack([ring, ring[:1]])
        ax.plot(ring[:, 0], ring[:, 1], color=color, linewidth=1.2,
                linestyle=linestyle, zorder=zorder,
                label=label if i == 0 else None)
        if scores is not None:
            ax.text(box[0], box[1], f'{scores[i]:.2f}', color=color, fontsize=5,
                    ha='center', va='center', zorder=zorder)


def pick_worst_frames(dump_path: str, n: int, drive: str | None = None) -> list[tuple[str, str]]:
    """N worst-recall frames among the highest-retention (top) quartile of
    warm frames, per a compare_results.py --dump per-frame json. Searches
    across all drives present in the dump, unless `drive` restricts it to
    one. Falls back to the full warm-frame pool if that quartile has fewer
    than `n` frames. Returns (drive, frame) pairs.
    """
    rows = json.load(open(dump_path))
    warm = [r for r in rows if not r['cold'] and r['n_base'] > 0]
    if drive is not None:
        warm = [r for r in warm if r['drive'] == drive]
    if not warm:
        scope = f'drive {drive!r}' if drive else 'any drive'
        raise SystemExit(f'no warm frames with baseline detections for {scope} in {dump_path}')

    warm.sort(key=lambda r: r['kept_fraction'])
    q_cut = int(len(warm) * 0.75)
    pool = warm[q_cut:] if len(warm) - q_cut >= n else warm
    pool = sorted(pool, key=lambda r: r['n_matched'] / r['n_base'])
    return [(r['drive'], r['frame']) for r in pool[:n]]


def pick_all_frames(baseline_dir: str, warm_dir: str, drive: str | None = None) -> list[tuple[str, str]]:
    """Every (drive, frame) common to both runs' manifests, sorted -- so
    --all-frames plots the full common set instead of a hand-picked subset.
    Restricted to one drive if `drive` is given.
    """
    base_manifest = load_manifest(Path(baseline_dir))
    warm_manifest = load_manifest(Path(warm_dir))
    common = set(base_manifest) & set(warm_manifest)
    if drive is not None:
        common = {k for k in common if k[0] == drive}
    if not common:
        scope = f'drive {drive!r}' if drive else 'any drive'
        raise SystemExit(f'no common baseline/warm frames for {scope} in '
                          f'{baseline_dir!r} / {warm_dir!r}')
    return sorted(common)


def plot_frame(args, drive: str, frame: str) -> Path:
    bin_path = (Path(args.kitti_root) / drive / 'velodyne_points'
                / 'data' / f'{frame}.bin')
    pts = load_points(bin_path, args.point_stride)

    b_boxes, b_scores = load_boxes(Path(args.baseline_dir) / drive / f'{frame}.json')
    w_boxes, w_scores = load_boxes(Path(args.warm_dir) / drive / f'{frame}.json')

    if args.min_score is not None:
        b_keep = b_scores >= args.min_score
        b_boxes, b_scores = b_boxes[b_keep], b_scores[b_keep]
        w_keep = w_scores >= args.min_score
        w_boxes, w_scores = w_boxes[w_keep], w_scores[w_keep]

    b_samples_path = Path(args.baseline_dir) / drive / f'{frame}_samples.npy'
    w_samples_path = Path(args.warm_dir) / drive / f'{frame}_samples.npz'

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c='#bbbbbb', linewidths=0,
               label='point cloud')

    draw_boxes(ax, w_boxes, '#d62728', 'warm-D-FPS detections',
               scores=w_scores if args.show_scores else None,
               linestyle='-', zorder=2)
    draw_boxes(ax, b_boxes, '#1f77b4', 'baseline detections',
               scores=b_scores if args.show_scores else None,
               linestyle='--', zorder=3)

    if b_samples_path.exists():
        Sb = np.load(b_samples_path)
        ax.scatter(Sb[:, 0], Sb[:, 1], s=0.3, marker='x', c='#1f77b4',
                   label='baseline D-FPS samples')
    else:
        print(f'note: no baseline samples at {b_samples_path} '
              f'(rerun with --dump-samples)')

    if w_samples_path.exists():
        d = np.load(w_samples_path)
        Sw, n_seed = d['S'], int(d['n_seed'])
        if n_seed > 0:
            ax.scatter(Sw[:n_seed, 0], Sw[:n_seed, 1], s=0.3, marker='^',
                       c='#ff7f0e', label='warm: carried seeds')
        ax.scatter(Sw[n_seed:, 0], Sw[n_seed:, 1], s=0.3, marker='+',
                   c='#2ca02c', label='warm: freshly refilled')
    else:
        print(f'note: no warm samples at {w_samples_path} '
              f'(rerun with --dump-samples)')

    ax.set_xlabel('x (m, forward)')
    ax.set_ylabel('y (m, left)')
    ax.set_aspect('equal')
    ax.set_title(f'{drive} / {frame} — BEV: baseline vs. warm-D-FPS')
    if args.xlim:
        ax.set_xlim(args.xlim)
    if args.ylim:
        ax.set_ylim(args.ylim)
    ax.legend(loc='upper right', fontsize=8, markerscale=1.5)
    fig.tight_layout()

    out = Path(args.out_dir) / f'bev_{drive}_{frame}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kitti-root', required=True)
    ap.add_argument('--baseline-dir', default='baseline_results')
    ap.add_argument('--warm-dir', default='warm_results')
    ap.add_argument('--drive', default=None,
                     help='Required with --frame. Optional with --num-frames: '
                          'restricts auto-picking to one drive instead of '
                          'searching across all drives in --per-frame-dump.')
    ap.add_argument('--frame', nargs='*', default=None,
                     help='One or more frame stems, e.g. 0000000042 (requires --drive). '
                          'Mutually exclusive with --num-frames.')
    ap.add_argument('--num-frames', type=int, default=None,
                     help='Auto-pick this many worst-recall frames from the '
                          'highest-retention (Q4) bucket, using --per-frame-dump. '
                          'Mutually exclusive with --frame and --all-frames.')
    ap.add_argument('--per-frame-dump', default='per_frame.json',
                     help='compare_results.py --dump output (needed with --num-frames)')
    ap.add_argument('--all-frames', action='store_true',
                     help='Plot every frame common to --baseline-dir and --warm-dir '
                          '(optionally restricted to one drive with --drive). '
                          'One PNG per frame -- slow/disk-heavy over a full drive. '
                          'Mutually exclusive with --frame and --num-frames.')
    ap.add_argument('--out-dir', default='.', help='Directory to write PNGs into')
    ap.add_argument('--point-stride', type=int, default=4,
                     help='Subsample the background point cloud for plotting speed')
    ap.add_argument('--xlim', type=float, nargs=2, default=None)
    ap.add_argument('--ylim', type=float, nargs=2, default=None)
    ap.add_argument('--show-scores', action='store_true',
                     help='Annotate each box (baseline and warm) with its confidence score (scores_3d)')
    ap.add_argument('--min-score', type=float, default=None,
                     help='Only draw boxes with score >= this value, on both baseline '
                          'and warm -- isolates the confident detections (test_cfg '
                          'dumps up to max_output_num regardless of score, so this '
                          'strips the low-confidence/noise-tier candidates)')
    args = ap.parse_args()

    if sum(bool(x) for x in (args.frame, args.num_frames, args.all_frames)) > 1:
        ap.error('--frame, --num-frames, and --all-frames are mutually exclusive')

    if args.num_frames:
        frames = pick_worst_frames(args.per_frame_dump, args.num_frames, drive=args.drive)
        scope = args.drive or 'all drives'
        print(f'Auto-picked {len(frames)} frame(s) from {scope} '
              f'(worst recall, top-retention quartile): {frames}')
    elif args.all_frames:
        frames = pick_all_frames(args.baseline_dir, args.warm_dir, drive=args.drive)
        scope = args.drive or 'all drives'
        print(f'Plotting all {len(frames)} common frame(s) from {scope}')
    elif args.frame:
        if not args.drive:
            ap.error('--frame requires --drive')
        frames = [(args.drive, f) for f in args.frame]
    else:
        ap.error('specify --frame (one or more), --num-frames, or --all-frames')

    for i, (drive, frame) in enumerate(frames, 1):
        out = plot_frame(args, drive, frame)
        print(f'[{i}/{len(frames)}] Wrote {out}')


if __name__ == '__main__':
    main()
