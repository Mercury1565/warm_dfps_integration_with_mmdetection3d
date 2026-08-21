# Warm-D-FPS Evaluation Pipeline

Tools for running baseline vs. warm-started D-FPS sampling inference on KITTI raw
drives, then comparing detection quality (frame-level, per-box, ego-motion
sensitivity, and BEV visualization).

Typical pipeline order:

1. `batch_baseline_inference.py`: run the baseline model, dump detections (+ optional D-FPS samples)
2. `warm_baseline_inference.py`: run the warm-started variant(s): plain / compensated / range-adaptive
3. `compare_results.py`: frame-level recall/precision, baseline vs. a warm variant
4. `analyze_motion_effect.py`: bucket recall by ego displacement, compensated vs. uncompensated
5. `box_correspondence.py` + `compare_correspondence.py`: per-box matching and match-rate reporting
6. `plot_bev.py` (or `generate_bev.sh`): bird's-eye-view plots of specific or auto-picked frames

---

## 1. `batch_baseline_inference.py`

Runs the baseline (cold-start) 3D detector over KITTI raw drives and dumps
per-frame detection JSONs (optionally with D-FPS sample positions).

**Usage**
```
python batch_baseline_inference.py \
    --kitti-root <path> \
    --config <config.py> \
    --checkpoint <checkpoint.pth> \
    --out-dir <output_dir> \
    [--device DEVICE] \
    [--drives DRIVE [DRIVE ...]] \
    [--overwrite] \
    [--dump-samples]
```

**Flags**

| Flag | Description |
|---|---|
| `--kitti-root` | Directory containing `*_sync` raw drive folders |
| `--config` | Model config file |
| `--checkpoint` | Model checkpoint file |
| `--out-dir` | Output directory for results |
| `--device` | Device to run inference on |
| `--drives` | Optional subset of drive folder names to run (e.g. `2011_09_26_drive_0001_sync`). Default: all |
| `--overwrite` | Recompute frames even if an output JSON already exists |
| `--dump-samples` | Also save each frame's SA1 D-FPS sample positions to `<out-dir>/<drive>/<frame>_samples.npy`, for BEV comparison against the warm-D-FPS run |

**Example**
```bash
python batch_baseline_inference.py \
    --kitti-root ../data/kitti/2011_09_26 \
    --config ../configs/3dssd/3dssd_4x4_kitti-3d-car.py \
    --checkpoint ../configs/3dssd/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth \
    --out-dir results/baseline_results --overwrite --dump-samples
```

---

## 2. `warm_baseline_inference.py`

Same detector, but carries sampled seed points forward between frames
("warm-started" D-FPS) instead of recomputing from scratch. Supports three
modes: no motion compensation, ego-motion-compensated (default), and
range-adaptive (compensation + per-sample local density thresholds).

**Usage**
```
python warm_baseline_inference.py \
    --kitti-root <path> \
    --config <config.py> \
    --checkpoint <checkpoint.pth> \
    --out-dir <output_dir> \
    [--device DEVICE] \
    [--drives DRIVE [DRIVE ...]] \
    [--overwrite] \
    [--dump-samples] \
    [--no-motion-compensation] \
    [--imu-to-velo-calib IMU_TO_VELO_CALIB]
```

**Flags**

| Flag | Description |
|---|---|
| `--kitti-root` | Directory containing `*_sync` raw drive folders |
| `--config` | Model config file |
| `--checkpoint` | Model checkpoint file |
| `--out-dir` | Output directory for results |
| `--device` | Device to run inference on |
| `--drives` | Optional subset of drive folder names to run. Default: all |
| `--overwrite` | Recompute frames even if an output JSON already exists |
| `--dump-samples` | Save each frame's SA1 sample positions and carried-seed count to `<out-dir>/<drive>/<frame>_samples.npz` (keys: `S`, `n_seed`), for BEV comparison against the baseline run |
| `--no-motion-compensation` | Disable oxts-based ego-motion compensation of carried samples (reproduces the original, uncompensated warm-D-FPS behavior) |
| `--imu-to-velo-calib` | Path to `calib_imu_to_velo.txt`. Default: `<kitti-root>/calib_imu_to_velo.txt` |

**Examples**

Plain warm (no motion compensation, flat threshold):
```bash
python warm_baseline_inference.py \
    --kitti-root ../data/kitti/2011_09_26 \
    --config ../configs/3dssd/3dssd_4x4_kitti-3d-car.py \
    --checkpoint ../configs/3dssd/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth \
    --out-dir results_3/warm_results \
    --no-motion-compensation \
    --no-range-adaptive \
    --overwrite \
    --dump-samples
```

Ego-motion compensated (default behavior):
```bash
python warm_baseline_inference.py \
    --kitti-root ../data/kitti/2011_09_26 \
    --config ../configs/3dssd/3dssd_4x4_kitti-3d-car.py \
    --checkpoint ../configs/3dssd/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth \
    --out-dir results_3/warm_results_compensated \
    --overwrite \
    --no-range-adaptive \
    --dump-samples
```

Range-adaptive (compensation + per-sample local density thresholds):
```bash
python warm_baseline_inference.py \
    --kitti-root ../data/kitti/2011_09_26 \
    --config ../configs/3dssd/3dssd_4x4_kitti-3d-car.py \
    --checkpoint ../configs/3dssd/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth \
    --out-dir results_3/warm_results_range_adaptive \
    --overwrite \
    --dump-samples
```

---

## 3. `compare_results.py`

Frame-level comparison of baseline vs. a warm-D-FPS results directory —
recall/precision aggregates, with optional per-frame JSON dump for
downstream tools.

**Usage**
```
python compare_results.py \
    --baseline-dir <dir> --warm-dir <dir> \
    [--iou-thr IOU_THR] [--score-thr SCORE_THR] [--dump DUMP]
```

**Flags**

| Flag | Description |
|---|---|
| `--baseline-dir` | Baseline results directory |
| `--warm-dir` | Warm-D-FPS results directory |
| `--iou-thr` | IoU threshold for matching |
| `--score-thr` | Drop detections below this score on each side before matching (`test_cfg` dumps up to `max_output_num` regardless of confidence, so `0.0` includes the full noise tail) |
| `--dump` | Optional path to dump per-frame results as JSON |

**Examples**

Unfiltered (includes full noise-tier tail, up to 100 raw candidates/frame):
```bash
python compare_results.py --baseline-dir results/baseline_results \
    --warm-dir results/warm_results_range_adaptive
```

Confident detections only (strips noise-floor artifacts):
```bash
python compare_results.py --baseline-dir results/baseline_results \
    --warm-dir results/warm_results_range_adaptive --score-thr 0.3
```

Dump per-frame recall/precision to JSON (feeds `analyze_motion_effect.py` and `plot_bev.py --num-frames`):
```bash
python compare_results.py --baseline-dir results/baseline_results --warm-dir results/warm_results \
    --score-thr 0.3 --dump report/per_frame_warm_filtered.json
# repeat per variant: warm_results_compensated, warm_results_range_adaptive
```

---


## 5. `box_correspondence.py`

Per-box baseline ↔ warm-D-FPS correspondence, dumped as CSV. Unlike
`compare_results.py` (frame-level aggregates only), this reports the
per-box detail: for every baseline box scoring ≥ `--score-thr`, it finds the
best-overlapping warm box (if any) and reports both confidence scores side
by side. Uses the same greedy same-class IoU matcher as `compare_results.py`
(highest IoU first, one-to-one, only pairs ≥ `--iou-thr` count), but matches
against the **full, unfiltered** warm box list — so a low-confidence warm
candidate that's still "the same car" shows up with its real score instead
of being invisible.

**Usage**
```
python box_correspondence.py \
    --baseline-dir <dir> --warm-dir <dir> \
    [--score-thr SCORE_THR] [--iou-thr IOU_THR] [--out OUT]
```

**Flags**

| Flag | Description |
|---|---|
| `--baseline-dir` | Baseline results directory |
| `--warm-dir` | Warm-D-FPS results directory |
| `--score-thr` | Only include baseline boxes with score ≥ this value |
| `--iou-thr` | Minimum IoU for a warm box to count as corresponding |
| `--out` | Output CSV path |

**Example**
```bash
python box_correspondence.py --baseline-dir results/baseline_results --warm-dir results/warm_results \
    --score-thr 0.3 --iou-thr 0.5 --out correspondence/baseline_vs_warm.csv
# repeat for warm_results_compensated, warm_results_range_adaptive
```

### `compare_correspondence.py`

Turns a `box_correspondence.py` CSV into a text report: overall match rate,
match rate by range bucket, match rate by refill fraction, and
worst-matched-pair lists.

**Usage**
```
python compare_correspondence.py <correspondence.csv> --out-dir <report_dir>
```

**Example**
```bash
python compare_correspondence.py correspondence/baseline_vs_warm.csv
```

---

## 6. `plot_bev.py`

Bird's-eye-view (BEV) plots comparing baseline vs. warm detections (and
point clouds) for specific frames, or auto-picked worst-recall frames from a
per-frame dump.

**Usage**
```
python plot_bev.py --kitti-root <path> \
    [--baseline-dir DIR] [--warm-dir DIR] \
    [--drive DRIVE] [--frame FRAME [FRAME ...]] \
    [--num-frames NUM_FRAMES] [--per-frame-dump DUMP] \
    [--out-dir OUT_DIR] [--point-stride N] \
    [--xlim MIN MAX] [--ylim MIN MAX] \
    [--show-scores] [--min-score MIN_SCORE]
```

**Flags**

| Flag | Description |
|---|---|
| `--kitti-root` | Directory containing `*_sync` raw drive folders |
| `--baseline-dir` | Baseline results directory |
| `--warm-dir` | Warm-D-FPS results directory |
| `--drive` | Required with `--frame`. Optional with `--num-frames`: restricts auto-picking to one drive instead of searching across all drives in `--per-frame-dump` |
| `--frame` | One or more frame stems, e.g. `0000000042` (requires `--drive`). Mutually exclusive with `--num-frames` |
| `--num-frames` | Auto-pick this many worst-recall frames from the highest-retention (Q4) bucket, using `--per-frame-dump`. Mutually exclusive with `--frame` |
| `--per-frame-dump` | `compare_results.py --dump` output (needed with `--num-frames`) |
| `--out-dir` | Directory to write PNGs into |
| `--point-stride` | Subsample the background point cloud for plotting speed |
| `--xlim` | X-axis plot limits (two floats) |
| `--ylim` | Y-axis plot limits (two floats) |
| `--show-scores` | Annotate each box (baseline and warm) with its confidence score (`scores_3d`) |
| `--min-score` | Only draw boxes with score ≥ this value, on both baseline and warm — isolates the confident detections (`test_cfg` dumps up to `max_output_num` regardless of score, so this strips the low-confidence/noise-tier candidates) |

**Examples**

Single explicit frame:
```bash
python plot_bev.py 
    --kitti-root ../data/kitti/2011_09_26 \
    --baseline-dir results/baseline_results \
    --warm-dir results/warm_results_range_adaptive \
    --drive 2011_09_26_drive_0017_sync \
    --frame 0000000087 \
    --show-scores --min-score 0.3 --out-dir bev_plots
```

Auto-pick N worst-recall frames from a per-frame dump:
```bash
python plot_bev.py 
    --kitti-root ../data/kitti/2011_09_26 \
    --baseline-dir results/baseline_results \
    --warm-dir results/warm_results_range_adaptive \
    --num-frames 10 \
    --per-frame-dump report/per_frame_warm_range_adaptive_filtered.json \ --out-dir bev_n_worst
```

```bash
python plot_bev.py
    --kitti-root ../data/kitti/2011_09_26 \
    --baseline-dir results/baseline_results \
    --warm-dir results/warm_results_range_adaptive \
    --all-frames --out-dir bev_plots --show-scores --min-score 0.3
```

---

## Notes

- `--score-thr` / `--min-score` matter because `test_cfg` dumps up to
  `max_output_num` detections per frame regardless of confidence — a
  threshold of `0.0` (or omitting the flag) includes the full noise-floor
  tail.
- The three warm variants (`warm_results`, `warm_results_compensated`,
  `warm_results_range_adaptive`) are typically run and compared in parallel
  through `compare_results.py`, `box_correspondence.py`, and `plot_bev.py`
  to see the incremental effect of motion compensation and range-adaptive
  thresholds.