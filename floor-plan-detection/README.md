# Floor-plan detection (vector-first, CPU)

Detects rooms and architectural elements in **vector PDF floor plans** (CAD exports). Tested on
`Student Accomodation.pdf` (A1, 1:100). See `analysis_report.md` for the approach, measured results and known limitations.
**Two modes, chosen automatically:** vector PDFs (CAD exports) use the vector pipeline; raster-only PDFs (an embedded picture, e.g. `floor_plan.pdf`, `floor_plan2.pdf`) use a classical raster pipeline (Otsu + wall-thickness mask, OCR labels as room seeds, geodesic partition, door arcs by circle fitting). The raster mode has **no trained room model**: it needs readable room labels and is weaker than the vector mode (see `review_v3.md`, section 5).

## Setup (tested: Windows 11, uv, CPython 3.11.16)
```powershell
cd floor-plan-detection
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```
The CPU install above is complete and runs on its own (~42 s, CPU-only). **Optional GPU acceleration** (NVIDIA only; the AMD Radeon 740M is not used):
```powershell
uv pip uninstall --python .venv\Scripts\python.exe onnxruntime
uv pip install --python .venv\Scripts\python.exe -r requirements-gpu.txt    # torch cu128 (~2.7 GB) + onnxruntime-gpu 1.23.2
```
`onnxruntime-gpu` must stay at **1.23.2** (newer builds target CUDA 13 and do not match PyTorch cu128). `--device auto` (default) uses CUDA when PyTorch sees it,
otherwise CPU; `--device cpu|cuda` force it (`cuda` fails loudly if unavailable). What runs on the GPU: OCR (RapidOCR via ONNX Runtime CUDA) and the symbol
template matching (PyTorch FFT). Measured on the RTX 2050: x1.39 pipeline speed-up (see `outputs/benchmark/`), peak VRAM ~1.7 GB of 4 GB, identical detection output.

## Run (single command)
```powershell
.venv\Scripts\python.exe -m src.cli "..\Student Accomodation.pdf" --out outputs
```
~30 s on CUDA / ~42 s CPU-only (laptop on battery; see benchmark), 1-2 GB RAM. Flags: `--device auto|cpu|cuda`, `--no-gpu-sampling`, `--no-ocr`, `--no-label-split` (ablation), `--config`, `--classes`.
`run_stats.json` records the device, which ONNX providers the OCR sessions really use, per-stage times and (when sampling) measured GPU utilisation / VRAM.
CPU-vs-GPU benchmark (interleaved rounds): `python -m src.evaluation.benchmark_device "..\Student Accomodation.pdf" <dir> --rounds 5` (needs `.venv_cpuref`, a CPU-only reference env).
Then, optionally: `.venv\Scripts\python.exe -m src.evaluation.report outputs` and `.venv\Scripts\python.exe -m pytest -q`.

## Latest (v5): adaptive vector pipeline, very large plans
`review_v5.md`: stroke threshold, plan footprint and pt-parameters are derived from the sheet (scaled by the printed scale); direction-agnostic wall-end extension; open-plan split by labels; circles as elements; school room classes; 6-12x faster path extraction; sheet profiles (`config/profiles/`, `--profile`, `run.bat ... [profile]`). Student unchanged (IoU 1.0), `expected.png` metrics unchanged, A1800 (A0, 1:200) 73 -> 259 rooms, 8 -> 35 of 38 labels inside a room. Tests: 68.

## Earlier (v4): accurate boundaries on simple plans
`review_v4.md` documents the root causes, changes, measured results and limits; `evaluation_v4/` holds the generated evaluation (`summary.md`, full-page comparison images against `expected.png`).
* Raster-only PDFs now use wall-end extension across openings, text suppression, exact outlines and shape descriptors (`src/raster/segmentation.py`); the vector path is unchanged (v3 vs v4 on the complex plan: min IoU 1.0, 0.0 pt displacement).
* Every room has `shape` (shape_type, rectangularity, solidity, circularity, axis-aligned edge fraction) and, in raster mode, coordinates in three frames: `polygon_image_px` (source image), `polygon_pt` (PDF points) and `polygon` (full-page render px).
* Reproduce: `python -m src.evaluation.run_v4_evaluation` then `python -m src.evaluation.make_review_v4`; tests: `python -m pytest tests -q` (58).

## Versions
* `outputs/`, `src/`, `config/` = **v3** (current: GPU support, full-page outputs, windows/annex pods). `versions/v3/` (also with the raster outputs before v4), `versions/v2/` and `versions/v1/` (each `{src,config,outputs}`) are the earlier versions; the v4 changes are the current `src/`; v1 reproducible with
  `cd versions\v1; ..\..\.venv\Scripts\python.exe -m src.cli "..\..\..\Student Accomodation.pdf" --out <dir>` (verified identical).
* Reviews + measured differences: `review_v3.md` (v2 -> v3: GPU, cropping, detection; `outputs/comparison_v2_vs_v3/`) and `review_v2.md` (v1 -> v2; `versions/v2/outputs/comparison/`).
  Regenerate: `python -m src.evaluation.compare_versions versions/v2/outputs outputs outputs/comparison_v2_vs_v3 v2 v3`.
* Extra checks: `python -m src.evaluation.named_spaces <dir>` (stair/riser/lifts/AOV/store/lobby/tutor flat),
  `python -m src.evaluation.bedroom_consistency <dir>`.

## Outputs (`outputs/`)
| File | Content |
|---|---|
| `overlay.png` | **entire page** (3576 x 2526 = exactly the page render, same aspect) with class-coloured rooms, boundaries, door hinges, windows, columns, legend inset; red ring = `review_required` |
| `overlay_plan_zoom_cropped.png` | extra: crop of the detection ROI from `overlay.png` (clearly a crop) |
| `room_masks.png` | class-colour segmentation mask, **entire page**, white background |
| `rooms.json` | one record per room (schema below) |
| `elements.json` | `door`, `window`, `column`, `wall` records |
| `evaluation.json`, `run_stats.json` | proxy metrics, geometry checks, runtime, peak RAM |
| `comparison_v2_vs_v3/`, `benchmark/` | version comparison with side-by-side panels; CPU-vs-GPU benchmark |
| `debug/` | `render_reference.png` (the full-page pixel frame), `wall_mask_roi.png` (ROI only), `ocr.json`, `discarded_fragments.json` |

### Coordinates
`polygon` is in **pixels of the page rendered at `image.transform.scale_px_per_pt` (1.5)**, origin top-left
(`debug/render_reference.png`, 3576 x 2526 = the whole page). The detection ROI (`image.detection_roi_pt`) only limits where detection runs; it never crops the outputs. `polygon_pt` is the same polygon in page points of the displayed
(rotation-applied) page; `px = pt * scale`. Real-world units are *not* stored; `approx_area_m2` assumes the printed 1:100.

### `rooms.json` room record
`id`, `type` (see `config/classes.yaml`), `polygon`, `polygon_pt`, `holes`, `area_pixels`, `approx_area_m2`,
`extra_polygons` / `extra_polygons_pt` (optional: small pockets adopted into the room, same class), `confidence` (always `null`), `confidence_source`, `heuristic_score` (rule strength / template correlation — **not** a probability),
`evidence[]`, `flags[]`, `unit_ids[]`, `text_inside[]`, `elongation`, `inscribed_radius_pt`, `review_required`, `source`.

## Configuration
* `config/classes.yaml` — canonical classes, overlay colours, OCR keyword mapping (first match wins).
* `config/inference.yaml` — render scale, plan ROI, **building footprint** (page points), wall/door/extension parameters,
  minimum room area, en-suite and window exemplar boxes, compute device, tread removal, structural pass (named stair/lift/riser/AOV), pocket adoption, optional label-guided split. ROI, footprint and exemplars are specific to
  this sheet; a new drawing needs its own (documented in the file).

## Troubleshooting
* A raster-only PDF prints `no vector geometry ... -> raster mode`; rooms are then only as good as the OCR labels (an unreadable label = no room seed; enclosed unlabelled space is reported as `unknown`). Open-plan areas without walls are split midway between labels and flagged `open_plan_boundary_by_label_proximity`.
* `uv pip install` is slow on large wheels (~190 MB); wait, or re-run — downloads are cached.
* Missing rooms at the sheet edge — they are outside `vector.footprint`; extend it.
* Rooms merged / split on another drawing — tune `vector.extend_walls`, `door_arc`, `min_inscribed_radius_pt`; inspect `debug/wall_mask.png`.

## Java/EJB integration
Run the CLI as a subprocess (`python -m src.cli <pdf> --out <dir>`), then read `rooms.json` / `elements.json`.
`src.pipeline.run_vector(pdf, cfg, classes, out_dir)` is the single Python entry point.

## Layout
`src/vector` (path extraction, walls, doors, regions, elements, symbol template), `src/postprocessing` (polygons,
classification, instance split), `src/supporting_signals/ocr.py`, `src/output`, `src/evaluation`, `tests/`.
