"""
Per-box baseline<->warm-D-FPS correspondence, dumped as CSV.

This gives the per-box detail behind those numbers: for every baseline
box scoring >= --score-thr, find its best-overlapping warm box (if any) and
report both confidence scores side by side, so you can inspect exactly which
individual detections agree, disagree, or have no counterpart at all.

Correspondence is the same greedy same-class IoU matcher compare_results.py
uses (highest IoU first, one-to-one, only pairs >= --iou-thr count) -- but
here it's run baseline-filtered vs. the FULL (unfiltered) warm box list, so a
low-confidence warm candidate that a human would still call "the same car"
shows up with its real score instead of being invisible.

Usage:
    python box_correspondence.py \
        --baseline-dir results/baseline_results \
        --warm-dir results/warm_results_range_adaptive \
        --score-thr 0.3 --iou-thr 0.5 \
        --out report/box_correspondence.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from mmdet3d.structures import LiDARInstance3DBoxes

from comparison_result import load_frame, load_manifest


def correspond_boxes(b_boxes, b_scores, b_labels, w_boxes, w_scores, w_labels, iou_thr=0.5):
    """One row per baseline box: its best same-class warm match (>= iou_thr),
    or an unmatched row if no warm box qualifies. Matching is greedy and
    one-to-one -- a warm box can back at most one baseline box, so if two
    baseline boxes both overlap the same warm box, only the higher-IoU one
    claims it.
    """
    n_b, n_w = len(b_scores), len(w_scores)
    rows = []
    if n_b == 0:
        return rows

    matched = {}  # baseline idx -> (warm idx, iou)
    if n_w > 0:
        b3d = LiDARInstance3DBoxes(torch.from_numpy(b_boxes), box_dim=7)
        w3d = LiDARInstance3DBoxes(torch.from_numpy(w_boxes), box_dim=7)
        overlaps = LiDARInstance3DBoxes.overlaps(b3d, w3d).numpy()
        same_cls = b_labels[:, None] == w_labels[None, :]
        overlaps = np.where(same_cls, overlaps, 0.0)

        pairs = [(overlaps[i, j], i, j) for i in range(n_b) for j in range(n_w)]
        pairs.sort(key=lambda t: t[0], reverse=True)

        used_w = set()
        for iou, i, j in pairs:
            if iou < iou_thr:
                break
            if i in matched or j in used_w:
                continue
            matched[i] = (j, float(iou))
            used_w.add(j)

    for i in range(n_b):
        row = dict(
            baseline_idx=i,
            baseline_score=float(b_scores[i]),
            baseline_x=float(b_boxes[i, 0]),
            baseline_y=float(b_boxes[i, 1]),
            baseline_z=float(b_boxes[i, 2]),
        )
        if i in matched:
            j, iou = matched[i]
            row.update(
                matched=True,
                warm_idx=j,
                warm_score=float(w_scores[j]),
                iou=iou,
                center_dist=float(np.linalg.norm(b_boxes[i, :3] - w_boxes[j, :3])),
                warm_x=float(w_boxes[j, 0]),
                warm_y=float(w_boxes[j, 1]),
                warm_z=float(w_boxes[j, 2]),
            )
        else:
            row.update(matched=False, warm_idx=None, warm_score=None, iou=0.0,
                       center_dist=None, warm_x=None, warm_y=None, warm_z=None)
        rows.append(row)
    return rows


FIELDNAMES = [
    'drive', 'frame', 'baseline_idx', 'baseline_score',
    'baseline_x', 'baseline_y', 'baseline_z',
    'matched', 'warm_idx', 'warm_score', 'iou', 'center_dist',
    'warm_x', 'warm_y', 'warm_z',
    'n_carried_over', 'n_freshly_refilled',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--baseline-dir', default='baseline_results')
    ap.add_argument('--warm-dir', default='warm_results')
    ap.add_argument('--score-thr', type=float, default=0.3,
                     help='Only include baseline boxes with score >= this value')
    ap.add_argument('--iou-thr', type=float, default=0.5,
                     help='Minimum IoU for a warm box to count as corresponding')
    ap.add_argument('--out', default='box_correspondence.csv')
    args = ap.parse_args()

    base_root = Path(args.baseline_dir)
    warm_root = Path(args.warm_dir)
    base_manifest = load_manifest(base_root)
    warm_manifest = load_manifest(warm_root)
    common_keys = sorted(set(base_manifest) & set(warm_manifest))

    all_rows = []
    for drive, frame in common_keys:
        b_boxes, b_scores, b_labels = load_frame(base_root / drive / f'{frame}.json')
        w_boxes, w_scores, w_labels = load_frame(warm_root / drive / f'{frame}.json')

        keep = b_scores >= args.score_thr
        b_boxes, b_scores, b_labels = b_boxes[keep], b_scores[keep], b_labels[keep]

        wm = warm_manifest[(drive, frame)]
        n_carried_over = wm.get('n_kept')
        n_carried = wm.get('n_carried')
        n_freshly_refilled = (n_carried - n_carried_over
                              if n_carried is not None and n_carried_over is not None else None)

        for row in correspond_boxes(b_boxes, b_scores, b_labels, w_boxes, w_scores, w_labels,
                                     args.iou_thr):
            row['drive'] = drive
            row['frame'] = frame
            row['n_carried_over'] = n_carried_over
            row['n_freshly_refilled'] = n_freshly_refilled
            all_rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    n_matched = sum(r['matched'] for r in all_rows)
    print(f'{len(all_rows)} baseline boxes (score >= {args.score_thr}) across '
          f'{len(common_keys)} frames -- {n_matched} matched ({n_matched / max(1, len(all_rows)):.3f}), '
          f'{len(all_rows) - n_matched} with no corresponding warm box')
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
