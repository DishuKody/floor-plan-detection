# v3: GPU investigation, full-page outputs, detection improvements

Reference: `Student Accomodation.pdf`. Versions are preserved side by side: `versions/v1/`, `versions/v2/` (each with `src/`, `config/`, `outputs/`) and the
current v3 (`src/`, `config/`, `outputs/`). Every number below is read from files in `outputs/` (`benchmark/`, `comparison_v2_vs_v3/`, `run_stats.json`,
`evaluation.json`, `named_spaces_eval.json`), not typed from memory.

## 1. GPU: what I found, what I did, what it buys

### State before (inspected, not assumed)
RTX 2050 present and healthy (driver 592.27, 4096 MiB, ~3.3 GB free, compute capability 8.6). **Nothing could use it**: no PyTorch, no CuPy, ONNX Runtime had only the CPU provider, OpenCV had 0 CUDA devices.
A stage profile of the CPU pipeline (22.7 s on a faster-clocked laptop state) showed where a GPU could matter: OCR 11.3 s (half the run), pod template matching 2.6 s; path extraction (5.1 s, pure Python), mask building and polygonisation are not GPU-able.

### What was installed (and why only this)
* `torch==2.11.0+cu128` from the official PyTorch index (needed for CUDA matching and to supply the CUDA 12.8 / cuDNN 9 DLLs). No torchvision/torchaudio.
* `onnxruntime-gpu==1.23.2`, replacing the CPU build. **Compatibility problem found and fixed:** the latest `onnxruntime-gpu` 1.31 is built for CUDA 13 (`cublas64_13.dll` not found) while the PyTorch wheel is CUDA 12.8; pinning 1.23.2 lets it reuse PyTorch's DLLs. The pypi CDN delivered ~180 KB/s here, so the 244 MB wheel was fetched from a mirror and **its SHA-256 was verified against the digest published on pypi.org (match)** before installing.
* A second, clean CPU-only environment `.venv_cpuref` (no PyTorch, CPU-only onnxruntime 1.31) exists **only as the fair CPU reference for the benchmark**; it can be deleted.

### What actually runs on the GPU (verified, not claimed)
| Stage | Where | Evidence |
|---|---|---|
| OCR detection, classification, recognition (RapidOCR) | CUDA via ONNX Runtime | `run_stats.json -> ocr_providers`: all three sessions report `CUDAExecutionProvider` (ORT silently falls back to CPU otherwise, so this is read from the live sessions) |
| Pod + window template matching | CUDA via PyTorch (FFT correlation, integral images) | CPU/GPU results identical (below) |
| Path extraction, wall mask, door/arc fitting, polygonisation, classification, rendering | CPU | pure Python / OpenCV; not suited to GPU |

### Measured (interleaved benchmark: 5 rounds x 3 configs, rotating order; `outputs/benchmark/`)
The laptop was **on battery** (Balanced plan, CPU ~2.5 GHz, browser/other apps running) and its speed drifted by roughly 50% during the session; I did not change power settings. Hence the interleaving.

| configuration | pipeline s, median (min-max) | process wall s | OCR stage s | pod+symbol matching s |
|---|---|---|---|---|
| CPU-only env (no PyTorch, onnxruntime 1.31) | 42.4 (41.1-54.5) | 43.9 | 15.5 | 4.7 |
| GPU env, `--device cpu` | 43.4 (41.9-50.0) | 44.8 | 16.0 | 4.8 |
| **GPU env, `--device cuda`** | **30.6 (29.8-37.3)** | **34.6** | **7.3** | **1.9** |

* **Speed-up CUDA vs CPU-only env: x1.39 pipeline, x1.27 including process start (PyTorch import), x2.12 on the OCR stage.** Modest because the pipeline is mostly CPU-bound.
* **GPU utilisation / VRAM** (separate `nvidia-smi`-sampled run, 44 samples @ 0.25 s; polling slows the process so it is never used for timing): baseline VRAM used 405 MB, **peak 1,674 MB (+1,269 MB of 4,096)**, peak utilisation 84 %, **mean 12 %**. A second sampled run (final output) gave peak 1,673 MB, peak 100 %, mean 20 %.
* **Detection quality is unchanged by the GPU:** CPU-only vs CUDA outputs have identical room count (109), identical room types and polygons, identical 62 windows / 105 doors, and identical OCR text (132 boxes, same set). The PyTorch correlation matches OpenCV's `TM_CCOEFF_NORMED` to 4.9e-7 on windows containing ink (near-blank windows differ, where OpenCV's value is meaningless and the pipeline discards them) with identical match centres.
* RAM: peak 2.1 GB with CUDA vs 1.0 GB CPU-only.

### Things that did not work / corrections
* First PyTorch implementation (direct `conv2d`, 3 large convolutions per template variant) made pod matching **48 s on GPU vs 2.6 s on CPU**; replaced by an FFT correlator that reuses the image FFT across all template variants (1.5-1.9 s).
* My theory that RapidOCR's hardcoded cuDNN `EXHAUSTIVE` search slowed OCR was **wrong**: `HEURISTIC` gave no difference (4.66 vs 4.60 s warm). The option exists in config but is disabled.
* My `nvidia-smi` poller itself inflates runtime, so timing runs use `--no-gpu-sampling`.
* The first sequential benchmark falsely suggested the GPU was *slower* (33.95 s warm) because CPU speed drifted between the CPU block and the GPU block; the interleaved design above replaced it.
* Not done: mixed precision (FP16). The template-matching numerator is fp32, window statistics are fp64; I did not test FP16 because it would risk the near-blank-window numerics for no measurable benefit at 1-2 s total.

## 2. Cropping fix
**Before (v2):** `overlay.png` was the plan ROI plus a 10 pt margin (2842 x 1470) with a legend strip appended (3172 x 1470); `room_masks.png` was the same crop. Title block, key plan, border, annex and grid bubbles were lost, the canvas aspect differed from the PDF page.
**Now:** detection ROI (`vector.roi`, config) and output canvas are separate. `overlay.png`, `room_masks.png` and `debug/render_reference.png` are all **3576 x 2526 = the whole page** (A1: 2384 x 1684 pt at 1.5 px/pt, same aspect, no resampling). The legend is an inset placed in the corner with least ink (top-left, 0.0 ink covered). The old cropped view is kept only as the clearly named extra `overlay_plan_zoom_cropped.png`. `rooms.json -> image` now records `output_canvas`, `detection_roi_pt`, `building_footprint_pt`, `legend`.
**Verified (tests `tests/test_fullpage_output.py`):** all three images equal the render size and the PDF page aspect; outside the painted rooms/elements/legend the overlay is **pixel-identical to the plain page render (0 of 9,032,976 pixels differ)** and the border rows/columns are identical, i.e. no cropping, shifting, stretching or resampling; each room's interior point is painted in the mask; JSON polygons `px = pt x 1.5` to 0.055 px and boundaries sit a mean 0.81 px from drawn ink.
Bug found by those tests: `unknown` rooms were coloured **white**, invisible in the mask and overlay; now dark grey.

## 3. Detection changes and what is (not) proven
| Change | Evidence | Status |
|---|---|---|
| Windows by template matching (north and south facade exemplars, rotated/flipped copies, 30 pt facade band) | windows 14 -> 62; visually checked on north, south and west-wing facades | **no ground truth for windows**; the east facade of the 5-bed wing still has none (drawn in a different style); a couple of overlapping duplicate boxes at the north-east end of the main block |
| Annex pods: 2 pods found as standalone bathrooms flagged `no_host_room:drawing_truncated_here` | bathrooms 39 -> 41; visually on the cut-off Core 2 fragment | the two annex **bedrooms stay undetected on purpose** (no walls are drawn; I will not invent polygons) |
| Unlabelled 3-10 m2 rooms touching a corridor are typed `corridor_or_hallway` (score 0.4, review-flagged, flag `type_by_adjacency_rule_unverified`) | unknown rooms 4 -> 2; 2 east-end lobbies re-typed | **unmeasured visual judgement** |
| Tutor Flat judged as a group | it is a flat with several rooms; the "single room 15-80 m2" check was ill-posed. Group of `other` rooms = 25.9 m2 (my own plausible range 25-80) | evaluation changed, not the detector |
| `unknown` colour fixed | tests | fix |

Measured v2 -> v3 (`outputs/comparison_v2_vs_v3/comparison.md`): rooms 107 -> 109, windows 14 -> 62, bathrooms 39 -> 41, `unknown` 4 -> 2, corridors 11 -> 13, standalone annex pods 0 -> 2; **unchanged**: bedroom units isolated+correct 37/39, kitchens 6/7, named spaces 7/8 (the one failing check is the ill-posed single-room Tutor Flat), 0 invalid polygons, 0 overlapping pairs, boundary-to-wall fit. **So v3 does not improve measured bedroom/named-space accuracy; it completes elements (windows, annex pods) and reduces unresolved rooms, without degrading anything I can measure.** The max wall-ink-inside-polygon rose to 0.113 because the two annex pod boxes contain the pods' own wall brackets (flagged, expected).
Runtime in that comparison (24.4 s v2 vs 32.6 s v3) is **not** comparable: different battery/thermal state, v3 adds window matching and a GPU sampler.

## 4. Still wrong / unverified
Annex bedrooms undetected (drawing truncated); east-facade windows of the 5-bed wing; window count unverified; lobby polygon still includes the strip between the cluster fire doors; bathroom polygons are symbol bounding boxes; 2 `unknown` rooms (a 1.6 m2 piece under the Riser, one east-end piece); 50 of 109 rooms flagged `review_required`; no manual ground truth, so IoU and boundary F-score remain **not established**.

## 5. Raster-only PDFs (`floor_plan.pdf`, `floor_plan2.pdf`) - added after the above
Until now these were rejected. Raster2Seq still cannot be built here, so I added a **classical raster mode** (no trained room model), selected automatically when a PDF has no vector geometry and an embedded image. Method: per-image Otsu threshold (the walls of `floor_plan.pdf` are dithered grey, a fixed threshold failed) -> closing + thickness opening = walls -> OCR (two upscales; GPU) gives room labels as seeds -> **geodesic nearest-seed partition** (walls impassable, door necks costly) -> type from the label text (priority list: "Kitchen veranda" is a balcony) -> door arcs by RANSAC circle fitting anchored on walls -> unlabelled enclosed space reported as `unknown`. Polygons are written in both frames (embedded-image pixels and full-page render pixels); overlay and mask are full-page.
What failed on the way: a watershed on the distance transform (kitchen flooded the whole house; labels placed off the basin minima) -> replaced by the geodesic partition; the first door detector fired on porch tiles and furniture -> now anchored on walls and stricter; the legend overlapped the logo -> placement searches for the emptiest region and scales with the page; room polygons had holes where bold text sat -> holes < 1.5 % of a room are filled.
Measured (`outputs`-style files in `output/` and `output2/`): `floor_plan.pdf` 20 rooms (3 bedrooms, 3 baths, 1 WC, kitchen, 2 verandas, 3 stores/cupboard, dining, living, 2 corridor/hall, 3 unknown), ~8 s; `floor_plan2.pdf` 7 rooms (2 bedrooms, bathroom, kitchen, dining, living, porch), ~5 s. **Cross-check against the drawings' own printed areas** (a consistency signal, not ground truth; scale = median ratio): `floor_plan.pdf` bedrooms -4.9 / -1.4 / +8.5 %, kitchen +5.0 %, living +1.5 %, covered veranda +7.8 %, kitchen veranda -19.6 %, **dining -62.5 %**; `floor_plan2.pdf` bedrooms -0.8 / +0.8 %, kitchen +6.3 %, and "dining 69 m2" which is an OCR misread of "3.0M x 2.3M".
Known problems: open-plan areas (dining/living/kitchen have no walls between them) are split midway between labels, so the dining table lands in the living room - flagged `open_plan_boundary_by_label_proximity`, 11 of the 20 rooms of `floor_plan.pdf` carry such a flag (14 of 20 are flagged `review_required` in total); the second `WC` of `floor_plan.pdf` was unreadable to OCR (read as "Iwd") and is reported as an unlabelled `unknown` room; door detection is partial (12 doors found in `floor_plan.pdf`, 3 in `floor_plan2.pdf`; I did not count the true number); windows and fixtures are not detected in raster mode; room boundaries near doors are rough (cone-shaped tips); no manual ground truth, so IoU is **not** established. 36 tests pass (incl. a raster end-to-end full-page test).
