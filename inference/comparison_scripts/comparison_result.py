from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mmdet3d.structures import LiDARInstance3DBoxes


def load_frame(path: Path):
    d = json.load(open(path))
    boxes = np.array(d['bboxes_3d'], dtype=np.float32).reshape(-1, 7)
    scores = np.array(d['scores_3d'], dtype=np.float32)
    labels = np.array(d['labels_3d'], dtype=np.int64)
    return boxes, scores, labels


def filter_by_score(boxes, scores, labels, thr):
    """Drop low-confidence candidates before matching. Both runs dump their
    full max_output_num list regardless of score (test_cfg's score_thr=0.0),
    so without this the comparison is dominated by noise-tier boxes that
    were never going to agree between two independent samplings in the
    first place. Filters each side on its own scores, independently.
    """
    keep = scores >= thr
    return boxes[keep], scores[keep], labels[keep]


def match_frame(b_boxes, b_scores, b_labels, w_boxes, w_scores, w_labels, iou_thr=0.5):
    n_b, n_w = len(b_scores), len(w_scores)
    result = dict(n_base=n_b, n_warm=n_w, n_matched=0, ious=[], score_diffs=[], center_dists=[])
    if n_b == 0 or n_w == 0:
        return result

    b3d = LiDARInstance3DBoxes(torch.from_numpy(b_boxes), box_dim=7)
    w3d = LiDARInstance3DBoxes(torch.from_numpy(w_boxes), box_dim=7)
    overlaps = LiDARInstance3DBoxes.overlaps(b3d, w3d).numpy()  # (n_b, n_w)

    same_cls = b_labels[:, None] == w_labels[None, :]
    overlaps = np.where(same_cls, overlaps, 0.0)

    centers_b = b_boxes[:, :3]
    centers_w = w_boxes[:, :3]

    pairs = [(overlaps[i, j], i, j) for i in range(n_b) for j in range(n_w)]
    pairs.sort(key=lambda t: t[0], reverse=True)

    used_b, used_w = set(), set()
    for iou, i, j in pairs:
        if iou < iou_thr:
            break
        if i in used_b or j in used_w:
            continue
        used_b.add(i)
        used_w.add(j)
        dist = float(np.linalg.norm(centers_b[i] - centers_w[j]))
        result['ious'].append(float(iou))
        result['score_diffs'].append(float(abs(b_scores[i] - w_scores[j])))
        result['center_dists'].append(dist)

    result['n_matched'] = len(result['ious'])
    return result


def load_manifest(out_dir: Path):
    m = json.load(open(out_dir / 'manifest.json'))
    return {(e['drive'], e['frame']): e for e in m}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-dir', default='baseline_results')
    ap.add_argument('--warm-dir', default='warm_results')
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--score-thr', type=float, default=0.0,
                     help='Drop detections below this score on each side before '
                          'matching (test_cfg dumps up to max_output_num regardless '
                          'of confidence, so 0.0 includes the full noise tail)')
    ap.add_argument('--dump', default=None, help='Optional path to dump per-frame results as JSON')
    args = ap.parse_args()

    base_root = Path(args.baseline_dir)
    warm_root = Path(args.warm_dir)

    base_manifest = load_manifest(base_root)
    warm_manifest = load_manifest(warm_root)

    common_keys = sorted(set(base_manifest) & set(warm_manifest))
    print(f'Common frames: {len(common_keys)} '
          f'(baseline-only: {len(set(base_manifest) - set(warm_manifest))}, '
          f'warm-only: {len(set(warm_manifest) - set(base_manifest))})')

    per_frame = []
    for drive, frame in common_keys:
        b_path = base_root / drive / f'{frame}.json'
        w_path = warm_root / drive / f'{frame}.json'
        b_boxes, b_scores, b_labels = load_frame(b_path)
        w_boxes, w_scores, w_labels = load_frame(w_path)
        if args.score_thr > 0.0:
            b_boxes, b_scores, b_labels = filter_by_score(b_boxes, b_scores, b_labels, args.score_thr)
            w_boxes, w_scores, w_labels = filter_by_score(w_boxes, w_scores, w_labels, args.score_thr)
        m = match_frame(b_boxes, b_scores, b_labels, w_boxes, w_scores, w_labels, args.iou_thr)
        m['drive'] = drive
        m['frame'] = frame
        m['cold'] = warm_manifest[(drive, frame)].get('cold', None)
        m['kept_fraction'] = warm_manifest[(drive, frame)].get('kept_fraction', 0.0)
        m['base_inference_s'] = base_manifest[(drive, frame)]['inference_s']
        m['warm_inference_s'] = warm_manifest[(drive, frame)]['inference_s']
        per_frame.append(m)

    if args.dump:
        with open(args.dump, 'w') as f:
            json.dump(per_frame, f, indent=2)
        print(f'Per-frame results dumped to {args.dump}')

    def summarize(rows, label):
        total_base = sum(r['n_base'] for r in rows)
        total_warm = sum(r['n_warm'] for r in rows)
        total_matched = sum(r['n_matched'] for r in rows)
        all_ious = [i for r in rows for i in r['ious']]
        all_score_diffs = [s for r in rows for s in r['score_diffs']]
        all_dists = [d for r in rows for d in r['center_dists']]

        print(f'\n--- {label} ({len(rows)} frames) ---')
        print(f'  baseline dets: {total_base}   warm dets: {total_warm}')
        if total_base:
            print(f'  recall (matched/base):    {total_matched / total_base:.3f}')
        if total_warm:
            print(f'  precision (matched/warm): {total_matched / total_warm:.3f}')
        if all_ious:
            print(f'  matched IoU:        mean={np.mean(all_ious):.3f}  min={np.min(all_ious):.3f}')
            print(f'  matched score diff: mean={np.mean(all_score_diffs):.4f}  max={np.max(all_score_diffs):.4f}')
            print(f'  matched center dist (m): mean={np.mean(all_dists):.3f}  max={np.max(all_dists):.3f}')

        base_t = [r['base_inference_s'] for r in rows]
        warm_t = [r['warm_inference_s'] for r in rows]
        if base_t:
            bt, wt = np.mean(base_t), np.mean(warm_t)
            print(f'  inference_s: baseline={bt:.4f}  warm={wt:.4f}  '
                  f'speedup={100 * (bt - wt) / bt:.1f}%')

    summarize(per_frame, 'ALL frames')
    summarize([r for r in per_frame if r['cold']], 'COLD frames (warm-run)')
    summarize([r for r in per_frame if not r['cold']], 'WARM frames (warm-run)')
    retention_analysis(per_frame)


def retention_analysis(rows):
    """How much of the previous frame's sample set survived into this frame's
    warm-D-FPS output (`kept_fraction`, from WarmStartManager.step), and
    whether frames that retain more of the previous frame agree with baseline
    more or less than frames that retain less.
    """
    warm_rows = [r for r in rows if not r['cold'] and r['n_base'] > 0]
    print(f'\n--- Retention vs. agreement ({len(warm_rows)} warm frames w/ baseline dets) ---')
    if not warm_rows:
        print('  nothing to analyze')
        return

    kfs = np.array([r['kept_fraction'] for r in warm_rows])
    recalls = np.array([r['n_matched'] / r['n_base'] for r in warm_rows])

    print(f'  kept_fraction (prev-frame samples surviving validity check): '
          f'mean={kfs.mean():.3f}  min={kfs.min():.3f}  max={kfs.max():.3f}')

    if kfs.std() > 0 and recalls.std() > 0:
        corr = float(np.corrcoef(kfs, recalls)[0, 1])
        print(f'  correlation(kept_fraction, per-frame recall): {corr:.3f}')

    order = np.argsort(kfs)
    for qi, idxs in enumerate(np.array_split(order, 4)):
        if len(idxs) == 0:
            continue
        print(f'    Q{qi + 1} kept_fraction {kfs[idxs].min():.3f}-{kfs[idxs].max():.3f} '
              f'(n={len(idxs)}): mean recall={recalls[idxs].mean():.3f}')


if __name__ == '__main__':
    main()
