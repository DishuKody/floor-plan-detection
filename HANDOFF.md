# HANDOFF - floor-plan detection project (read this first on a new machine / new session)

Goal (from `floor_plan_detection.md`, the original brief in this folder): detect and segment rooms and architectural elements in floor-plan PDFs, produce
overlays / masks / JSON polygons, evaluate honestly, keep it integrable with a Java/EJB backend (call the CLI, read `rooms.json` / `elements.json`).
State: **v8 is current and verified** (80 tests pass; v8 = independent area segmentation of A1800 against the annotated image, see `floor-plan-detection/review_v8.md`; outputs `areas_overlay.png` + `areas.json` give every area a unique ID and colour; v7 notes in review_v7.md. A1800 room-level IoU vs annotation: mean 0.52 -> 0.65, rooms IoU>=0.5 61% -> 76%, merged rooms 17 -> 11 of 49. v6: v6 = A1800 refinement against the hand-annotated reference, see `floor-plan-detection/review_v6.md`; v5: v5 adds sheet calibration, profiles, fast extraction and the A1800 large-plan work - read `floor-plan-detection/review_v5.md` first). Earlier versions are preserved in `floor-plan-detection/versions/v1..v4`.

## 1. Set up on another machine (Windows)
1. Copy this folder (the zip excludes `.venv*`, which are machine-specific). Keep the layout: `setup.bat`, `run.bat`, the PDFs, `expected.png` next to `floor-plan-detection\`.
2. `setup.bat` (CPU, works anywhere) or `setup.bat gpu` (adds PyTorch cu128 + onnxruntime-gpu 1.23.2 if `nvidia-smi` sees an NVIDIA GPU; ~3 GB).
   It installs `uv` if missing, creates `floor-plan-detection\.venv` (Python 3.11), installs `requirements.txt`, then `rapidocr-onnxruntime==1.4.4` with `--no-deps`
   (it declares the non-headless opencv-python, which clashes with the headless build), and self-checks.
3. Run: `run.bat "plan.pdf" "output_folder" [auto|cpu|cuda] [profile]` (profile = name in `config\profiles` or a yaml path: `student_accommodation`, `a1800_school`; vector sheets that are not 1:100 need one, because the printed scale is not stored as text) - relative paths count from where you run it. Result files are prefixed with the PDF name:
   `<name>_overlay.png`, `<name>_masks.png`, `<name>_rooms.json`, `<name>_elements.json` (+ `run_stats.json`, `debug/`).
4. Linux / macOS: not tested. `python -m src.cli "plan.pdf" --out out` from `floor-plan-detection` should work after creating a venv from `requirements.txt`.
No GPU needed: `--device auto` (default) uses CUDA only if PyTorch sees it; otherwise CPU. Verified: CPU-only copy, all 3 PDFs + 58 tests OK.

## 2. What the pipeline does (adaptive)
* **Vector PDF** (CAD export, e.g. `Student Accomodation.pdf`): PyMuPDF paths -> wall mask (stroke-width classes, grid dash removal, door arcs by circle fit with fragment merge,
  dangling-wall extension, door chords) -> free-space regions -> OCR labels / unit IDs, pod + window template matching (CAD blocks), structural pass for stair/lift/riser/AOV,
  rule-based classification with evidence. v5: stroke threshold, plan footprint/ROI and pt-parameters (scaled by 100/scale_ratio) are derived from the sheet; **sheet-specific facts** (printed scale, symbol exemplars) are in `config/profiles/*.yaml`; direction-agnostic wall-end extension, open-plan split by labels, circles as elements.
* **Raster-only PDF** (embedded picture, e.g. `floor_plan.pdf`): `src/raster/` - OCR (all text removed from wall mask; labels = seeds) -> exact wall mask -> wall-end extension across
  openings -> wall-cluster outline -> enclosed free space = rooms (open-plan split by label proximity, flagged) -> pixel-exact polygons, axis snapping, shape descriptors, door arcs, circles.
* The dispatcher is `run_vector` in `src/pipeline.py` (no vector paths + an embedded image -> raster).

## 3. Repo map (`floor-plan-detection/`)
`src/cli.py` entry | `src/pipeline.py` vector pipeline | `src/raster/{pipeline_raster,segmentation}.py` raster | `src/vector/*` paths, doors, walls, regions, template matching |
`src/postprocessing/{classify,polygons,instances,structural,shapes}.py` | `src/supporting_signals/ocr.py` | `src/gpu.py` device + nvidia-smi sampler | `src/evaluation/*` all metrics |
`config/{inference,classes}.yaml`, `config/profiles/` | `tests/` (76) | `versions/` v1-v3 | `evaluation_v4/` generated evaluation | `outputs/` Student results.

## 4. Documents (all `.md` are meant to be kept)
`HANDOFF.md` (this) | `floor_plan_detection.md` original brief | `floor-plan-detection/README.md` usage + schema | `analysis_report.md` input analysis + approach choice |
`review_v2.md` (v1->v2 core fixes), `review_v3.md` (GPU benchmark, full-page outputs, windows, raster mode), `review_v6.md` (annotated-reference evaluation, dash chains, glazed screens, polygon bug), `review_v5.md` (A1800 / adaptive vector pipeline, measured results), `review_v4.md` (shape boundaries vs `expected.png`; generated:
`python -m src.evaluation.make_review_v4`). `evaluation_v4/summary.md` = generated numbers.

## 5. Key measured facts (do not re-derive; re-measure if the code changes)
* `expected.png` = screenshot of the v3 `floor_plan.pdf` overlay + 13 hand-drawn red boundary lines + 2 red X (title, logo = not shapes). Sparse: recall measurable, precision not.
  v3 -> v4 mean distance of the lines to detected boundaries 21.1 -> 5.3 px; recall@8px 40 % -> 84 %; 9/13 lines >= 80 % (was 1). Lines 2 and 8 (hand-drawn slanted) got WORSE.
* Complex plan v3 -> v4: identical (109 rooms, min IoU 1.0, 0.0 pt displacement).
* Student sheet proxy: 37/39 bedrooms isolated+correct (unit-ID OCR labels, not manual ground truth); 7/8 named spaces (Tutor Flat check is ill-posed: it is several rooms).
* GPU (RTX 2050, 4 GB): x1.39 vector pipeline, x1.34 / x1.13 raster; ~1.7 GB VRAM; identical outputs to CPU. Laptop on battery -> timings drift; benchmarks are interleaved.
* No manual ground truth exists for IoU / boundary F1 beyond the 13 lines and printed room areas.

* v5 (A1800, 1:200, A0): v4 found 73 rooms / 8 of 38 labels; v5 259 rooms / 35 of 38; boundary-on-structural-stroke 0.43 -> 0.93; GPU x1.30 pipeline, x2.54 OCR; Student unchanged (IoU 1.0 vs v4); expected.png metrics unchanged. See `review_v5.md` section 3.

* v6 (A1800 vs `A1800 Ground Flr_expected.png`): line pixels separating two rooms 0.51 -> 0.59 (10 px offset; both-sides-in-rooms 0.41 -> 0.44), distance recall 0.88 -> 0.905, rooms > 150 m2 5 -> 1; Student IoU 1.0, raster plans identical. Corridor-side open edges are still mostly unmatched. See `review_v6.md`.

## 6. Pitfalls already hit (so you do not repeat them)
* `region_polygon` once lost every region whose contour pinches (nested GeometryCollection from `make_valid`): fixed with `largest_polygon`; use `debug.probe_points_pt` in a profile to see why a region at a page point is kept or dropped.
* The dash-chain rule matches regular window / door-sill runs on Student: keep it off in that profile. Thin clutter-weight lines are added AFTER the wall-end extension (before, they changed the skeleton and re-merged rooms).
* Judge reference agreement with the separation metric, not distance recall alone (a dense polygon mesh scores well on recall).
* Windows Application Control occasionally blocks a DLL import (shapely, PIL `_imaging`) at random: simply re-run the command.
* PDFs here have no text layer (outlined text): scale and labels cannot be read from the PDF; OCR is the only text source.
* Keyword classification: short keywords must be whole words (`wc` vs "Showcase"); never add substring keywords like `duct`.
* `onnxruntime-gpu` >= 1.31 targets CUDA 13; with PyTorch cu128 use `onnxruntime-gpu==1.23.2`. Never keep CPU `onnxruntime` and `onnxruntime-gpu` together.
* pypi CDN can be very slow; a mirror download must be verified against the SHA-256 on pypi.org.
* Otsu returns 0 on pure 0/255 images (handled). OpenCV closing treats the image border as "set" (pad first - handled).
* `robocopy /XD output` also excludes `src\output` (a real package) - exclude by full path.
* Bold text survives a wall-thickness filter: suppress via OCR boxes. Full-thickness door closures were worse than centreline closures against `expected.png`.
* Windows PowerShell blocks `.ps1` (execution policy): use `run.bat` / `setup.bat`.
* Do not change the user's power plan; do not claim GPU use without the `ocr_providers` / `gpu_measured` fields in `run_stats.json`.

## 7. Open problems / next steps (smallest first)
1. Hand-annotate ~15 rooms per plan to get real IoU / boundary F1 (only 13 hand-drawn lines exist).
2. Slanted / non-axis walls: evaluate `raster.rectilinear_walls: false`.
3. Open-plan splits (entrance / living / corridor) are curved label-proximity cuts: needs a better criterion than label distance.
4. Second WC in `floor_plan.pdf` unread by OCR; sink unit in `floor_plan2.pdf` reported as `unknown`; two oversized false circles in the WC.
5. Vector mode derives rooms from a 4 px/pt raster of vector walls (0.25 pt), not exact vector polygons.
6. Raster2Seq / learned models: not built here (no nvcc / MSVC); would need annotated plans first.

## 8. Working rules the user asked for (keep following)
Inspect before changing; run and look at the real outputs; never fabricate results or claim GPU/accuracy without measurement; preserve old versions to compare;
keep full-page outputs (never crop / stretch); do not hard-code per-PDF results or modify `expected.png`; report limits honestly.

## 9. Prompt to continue in a new Claude Code session
"Read HANDOFF.md, floor_plan_detection.md, floor-plan-detection/review_v4.md and review_v5.md. The project is at v5 (verified). Run setup.bat, then
`python -m pytest tests -q` and `python -m src.evaluation.run_v4_evaluation` inside floor-plan-detection to confirm the baseline, then work on: <your next goal>."
