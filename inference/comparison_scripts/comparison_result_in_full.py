"""
Baseline-vs-warm comparison across all three warm-D-FPS variants (plain,
motion-compensated, range-adaptive) and a sweep of score thresholds, written
out as CSVs instead of a single-run terminal report.

Reuses compare_results.py's matching logic (load_frame, filter_by_score,
match_frame, load_manifest) unchanged -- this script only adds the
variant x threshold sweep and CSV output on top.

Usage:
python generate_comparison_report.py \
    --baseline-dir results/baseline_results \
    --warm-dir results/warm_results \
    --warm-compensated-dir results/warm_results_compensated \
    --warm-range-adaptive-dir results/warm_results_range_adaptive \
    --out-dir comparison_report
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from inference.comparison_scripts.comparison_result import filter_by_score, load_frame, load_manifest, match_frame

SUMMARY_FIELDS = [
    'variant', 'score_thr', 'frame_subset', 'n_frames', 'n_base_dets', 'n_warm_dets',
    'n_matched', 'recall', 'precision', 'iou_mean', 'iou_min', 'score_diff_mean',
    'score_diff_max', 'center_dist_mean_m', 'center_dist_max_m',
    'base_inference_s_mean', 'warm_inference_s_mean', 'speedup_pct',
]
RETENTION_FIELDS = [
    'variant', 'score_thr', 'n_warm_frames', 'kept_fraction_mean',
    'kept_fraction_min', 'kept_fraction_max', 'corr_kept_fraction_recall',
]
RETENTION_QUARTILE_FIELDS = [
    'variant', 'score_thr', 'quartile', 'kept_fraction_min', 'kept_fraction_max',
    'n_frames', 'mean_recall',
]


def load_variant_frames(base_manifest: dict, base_root: Path, warm_root: Path) -> list[dict]:
    """Load every common frame's raw (unfiltered) boxes/scores/labels once per
    variant, so the threshold sweep re-filters and re-matches in memory
    instead of re-reading JSON off disk for each of the 4 thresholds.
    """
    warm_manifest = load_manifest(warm_root)
    common_keys = sorted(set(base_manifest) & set(warm_manifest))
    frames = []
    for drive, frame in common_keys:
        b_boxes, b_scores, b_labels = load_frame(base_root / drive / f'{frame}.json')
        w_boxes, w_scores, w_labels = load_frame(warm_root / drive / f'{frame}.json')
        frames.append(dict(
            drive=drive, frame=frame,
            b_boxes=b_boxes, b_scores=b_scores, b_labels=b_labels,
            w_boxes=w_boxes, w_scores=w_scores, w_labels=w_labels,
            cold=warm_manifest[(drive, frame)].get('cold', None),
            kept_fraction=warm_manifest[(drive, frame)].get('kept_fraction', 0.0),
            base_inference_s=base_manifest[(drive, frame)]['inference_s'],
            warm_inference_s=warm_manifest[(drive, frame)]['inference_s'],
        ))
    return frames


def match_at_threshold(frames: list[dict], score_thr: float, iou_thr: float) -> list[dict]:
    rows = []
    for fr in frames:
        b_boxes, b_scores, b_labels = fr['b_boxes'], fr['b_scores'], fr['b_labels']
        w_boxes, w_scores, w_labels = fr['w_boxes'], fr['w_scores'], fr['w_labels']
        if score_thr > 0.0:
            b_boxes, b_scores, b_labels = filter_by_score(b_boxes, b_scores, b_labels, score_thr)
            w_boxes, w_scores, w_labels = filter_by_score(w_boxes, w_scores, w_labels, score_thr)
        m = match_frame(b_boxes, b_scores, b_labels, w_boxes, w_scores, w_labels, iou_thr)
        m['cold'] = fr['cold']
        m['kept_fraction'] = fr['kept_fraction']
        m['base_inference_s'] = fr['base_inference_s']
        m['warm_inference_s'] = fr['warm_inference_s']
        rows.append(m)
    return rows


def summarize_stats(rows: list[dict]) -> dict:
    total_base = sum(r['n_base'] for r in rows)
    total_warm = sum(r['n_warm'] for r in rows)
    total_matched = sum(r['n_matched'] for r in rows)
    all_ious = [i for r in rows for i in r['ious']]
    all_score_diffs = [s for r in rows for s in r['score_diffs']]
    all_dists = [d for r in rows for d in r['center_dists']]
    base_t = [r['base_inference_s'] for r in rows]
    warm_t = [r['warm_inference_s'] for r in rows]

    base_mean = float(np.mean(base_t)) if base_t else None
    warm_mean = float(np.mean(warm_t)) if warm_t else None

    return dict(
        n_frames=len(rows),
        n_base_dets=total_base,
        n_warm_dets=total_warm,
        n_matched=total_matched,
        recall=(total_matched / total_base) if total_base else None,
        precision=(total_matched / total_warm) if total_warm else None,
        iou_mean=float(np.mean(all_ious)) if all_ious else None,
        iou_min=float(np.min(all_ious)) if all_ious else None,
        score_diff_mean=float(np.mean(all_score_diffs)) if all_score_diffs else None,
        score_diff_max=float(np.max(all_score_diffs)) if all_score_diffs else None,
        center_dist_mean_m=float(np.mean(all_dists)) if all_dists else None,
        center_dist_max_m=float(np.max(all_dists)) if all_dists else None,
        base_inference_s_mean=base_mean,
        warm_inference_s_mean=warm_mean,
        speedup_pct=(100 * (base_mean - warm_mean) / base_mean)
                    if base_mean else None,
    )


def retention_stats(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Mirrors compare_results.retention_analysis, returning data instead of
    printing it: how much of the previous frame's samples survived
    (kept_fraction) and whether that correlates with per-frame recall.
    """
    warm_rows = [r for r in rows if not r['cold'] and r['n_base'] > 0]
    if not warm_rows:
        return dict(n_warm_frames=0, kept_fraction_mean=None,
                     kept_fraction_min=None, kept_fraction_max=None, corr=None), []

    kfs = np.array([r['kept_fraction'] for r in warm_rows])
    recalls = np.array([r['n_matched'] / r['n_base'] for r in warm_rows])
    corr = (float(np.corrcoef(kfs, recalls)[0, 1])
            if kfs.std() > 0 and recalls.std() > 0 else None)

    summary = dict(
        n_warm_frames=len(warm_rows),
        kept_fraction_mean=float(kfs.mean()),
        kept_fraction_min=float(kfs.min()),
        kept_fraction_max=float(kfs.max()),
        corr=corr,
    )

    quartiles = []
    order = np.argsort(kfs)
    for qi, idxs in enumerate(np.array_split(order, 4)):
        if len(idxs) == 0:
            continue
        quartiles.append(dict(
            quartile=f'Q{qi + 1}',
            kept_fraction_min=float(kfs[idxs].min()),
            kept_fraction_max=float(kfs[idxs].max()),
            n_frames=int(len(idxs)),
            mean_recall=float(recalls[idxs].mean()),
        ))
    return summary, quartiles


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: ('' if row.get(k) is None else row[k]) for k in fieldnames})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-dir', default='baseline_results')
    ap.add_argument('--warm-dir', default='warm_results',
                     help='Plain warm-D-FPS results (no compensation, flat threshold)')
    ap.add_argument('--warm-compensated-dir', default='warm_results_compensated',
                     help='Motion-compensated warm-D-FPS results')
    ap.add_argument('--warm-range-adaptive-dir', default='warm_results_range_adaptive',
                     help='Motion-compensated + range-adaptive warm-D-FPS results')
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--score-thresholds', type=float, nargs='+',
                     default=[0.25, 0.30, 0.35, 0.40])
    ap.add_argument('--out-dir', default='comparison_report')
    args = ap.parse_args()

    base_root = Path(args.baseline_dir)
    base_manifest = load_manifest(base_root)

    variants = [
        ('warm', Path(args.warm_dir)),
        ('warm_compensated', Path(args.warm_compensated_dir)),
        ('warm_range_adaptive', Path(args.warm_range_adaptive_dir)),
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, retention_rows, retention_quartile_rows = [], [], []

    for label, warm_root in variants:
        print(f'{label}: loading {warm_root} ...')
        frames = load_variant_frames(base_manifest, base_root, warm_root)
        print(f'  {len(frames)} common frames')

        for score_thr in args.score_thresholds:
            rows = match_at_threshold(frames, score_thr, args.iou_thr)

            for subset_name, subset_rows in (
                ('all', rows),
                ('cold', [r for r in rows if r['cold']]),
                ('warm', [r for r in rows if not r['cold']]),
            ):
                stats = summarize_stats(subset_rows)
                summary_rows.append({
                    'variant': label, 'score_thr': score_thr,
                    'frame_subset': subset_name, **stats,
                })

            ret_summary, ret_quartiles = retention_stats(rows)
            retention_rows.append({
                'variant': label, 'score_thr': score_thr,
                'n_warm_frames': ret_summary['n_warm_frames'],
                'kept_fraction_mean': ret_summary['kept_fraction_mean'],
                'kept_fraction_min': ret_summary['kept_fraction_min'],
                'kept_fraction_max': ret_summary['kept_fraction_max'],
                'corr_kept_fraction_recall': ret_summary['corr'],
            })
            for q in ret_quartiles:
                retention_quartile_rows.append({'variant': label, 'score_thr': score_thr, **q})

    write_csv(out_dir / 'summary.csv', SUMMARY_FIELDS, summary_rows)
    write_csv(out_dir / 'retention.csv', RETENTION_FIELDS, retention_rows)
    write_csv(out_dir / 'retention_quartiles.csv', RETENTION_QUARTILE_FIELDS, retention_quartile_rows)
    print(f'Wrote {out_dir}/summary.csv, retention.csv, retention_quartiles.csv')


if __name__ == '__main__':
    main()
