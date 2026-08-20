"""
Usage:
    python compare_correspondence.py baseline_vs_warm_range_adaptive.csv
    python compare_correspondence.py baseline_vs_warm.csv --out-dir correspondence_report
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def _opt_float(v):
    return None if v in (None, '') else float(v)


def _opt_int(v):
    return None if v in (None, '') else int(float(v))


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                drive=r['drive'], frame=r['frame'], baseline_idx=int(r['baseline_idx']),
                baseline_score=float(r['baseline_score']),
                baseline_x=float(r['baseline_x']), baseline_y=float(r['baseline_y']),
                matched=r['matched'] == 'True',
                iou=float(r['iou']),
                center_dist=_opt_float(r['center_dist']),
                warm_score=_opt_float(r['warm_score']),
                n_carried_over=_opt_int(r.get('n_carried_over')),
                n_freshly_refilled=_opt_int(r.get('n_freshly_refilled')),
            ))
    return rows


def label_for(path: Path) -> str:
    stem = path.stem
    return stem[len('baseline_vs_'):] if stem.startswith('baseline_vs_') else stem


def _bucket(items: list, key, bins: int) -> list[list]:
    """Split ``items`` into ``bins`` roughly-equal groups, sorted by ``key``."""
    order = sorted(items, key=key)
    edges = np.linspace(0, len(order), bins + 1).round().astype(int)
    return [order[edges[i]:edges[i + 1]] for i in range(bins)]


def build_report(label: str, rows: list[dict], bins: int) -> str:
    lines = []
    pr = lines.append

    n = len(rows)
    matched = [r for r in rows if r['matched']]
    ious = [r['iou'] for r in matched]
    dists = [r['center_dist'] for r in matched if r['center_dist'] is not None]
    carried = [r['n_carried_over'] for r in rows if r['n_carried_over'] is not None]
    refilled = [r['n_freshly_refilled'] for r in rows if r['n_freshly_refilled'] is not None]

    pr(f'=== {label} ({n} baseline boxes) ===')
    pr(f'matched: {len(matched)} ({len(matched) / max(1, n):.3f})')
    if ious:
        pr(f'matched IoU:         mean={np.mean(ious):.3f}  min={np.min(ious):.3f}')
    if dists:
        pr(f'matched center dist: mean={np.mean(dists):.3f}  max={np.max(dists):.3f}')
    if carried:
        pr(f'n_carried_over:      mean={np.mean(carried):.1f}')
    if refilled:
        pr(f'n_freshly_refilled:  mean={np.mean(refilled):.1f}')

    # -- match rate by distance from ego --------------------------------------
    for r in rows:
        r['_range'] = math.hypot(r['baseline_x'], r['baseline_y'])

    pr('\n--- Match rate by distance from ego ---')
    pr(f'{"range bucket (m)":>20}  {"n":>5}  {"match rate":>10}')
    for bucket in _bucket(rows, key=lambda r: r['_range'], bins=bins):
        if not bucket:
            continue
        lo, hi = bucket[0]['_range'], bucket[-1]['_range']
        rate = np.mean([r['matched'] for r in bucket])
        pr(f'{lo:8.1f}-{hi:<10.1f}  {len(bucket):5d}  {rate:10.3f}')

    # -- match rate by how much of the budget was freshly refilled ------------
    refill_rows = [r for r in rows if r['n_freshly_refilled'] is not None]
    if refill_rows:
        pr('\n--- Match rate by n_freshly_refilled (this frame) ---')
        pr(f'{"refilled bucket":>20}  {"n":>5}  {"match rate":>10}')
        for bucket in _bucket(refill_rows, key=lambda r: r['n_freshly_refilled'], bins=bins):
            if not bucket:
                continue
            lo, hi = bucket[0]['n_freshly_refilled'], bucket[-1]['n_freshly_refilled']
            rate = np.mean([r['matched'] for r in bucket])
            pr(f'{lo:8d}-{hi:<10d}  {len(bucket):5d}  {rate:10.3f}')

    # -- worst offenders --------------------------------------------------------
    unmatched = sorted((r for r in rows if not r['matched']),
                       key=lambda r: -r['_range'])
    if unmatched:
        pr('\n--- Furthest unmatched baseline boxes (no corresponding warm box) ---')
        for r in unmatched[:10]:
            pr(f'  {r["drive"]}/{r["frame"]} baseline_idx={r["baseline_idx"]}  '
               f'range={r["_range"]:.1f}m  score={r["baseline_score"]:.2f}')

    worst_matched = sorted((r for r in matched if r['center_dist'] is not None),
                           key=lambda r: -r['center_dist'])
    if worst_matched:
        pr('\n--- Worst matched pairs (largest center distance) ---')
        for r in worst_matched[:10]:
            pr(f'  {r["drive"]}/{r["frame"]} baseline_idx={r["baseline_idx"]}  '
               f'range={r["_range"]:.1f}m  center_dist={r["center_dist"]:.2f}  iou={r["iou"]:.2f}')

    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv_path', help='box_correspondence.py output CSV to report on')
    ap.add_argument('--bins', type=int, default=4, help='Number of buckets for the breakdowns')
    args = ap.parse_args()

    path = Path(args.csv_path)
    label = label_for(path)
    rows = load_csv(path)
    report = build_report(label, rows, args.bins)

    print(report)

if __name__ == '__main__':
    main()
