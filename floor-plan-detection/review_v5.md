# v5: one pipeline for simple raster plans, complex CAD plans and a very large (A0, 1:200) school plan

All numbers below were measured in this session (files in `../evaluation_v5/` and `../evaluation_v5_runs/`); `python -m pytest tests -q` = **68 passed** (58 earlier + 10 new in `tests/test_vector_v5.py`).
v4 is preserved in `versions/v4/`. Deliverables: `../output_v5/<plan>/` and `../evaluation_v5/` (JSON shape coordinates, full-page overlays, masks, comparison images).

## 1. What was wrong with v4 on `A1800 Ground Flr.pdf` (root causes, each checked before changing code)
| # | root cause | evidence |
|---|---|---|
| 1 | The configuration was one sheet's profile: plan ROI, footprint rectangles, pod/window exemplars and `min_wall_stroke 0.2` were Student-Accommodation values. | v4 on A1800 looked only in a 1875x960 pt window of a 3158x2234 pt sheet; 8 of 38 room labels fell inside any room. |
| 2 | Pt-parameters (door radius 12-40, channel close 3.5, inscribed radius 7, extension 40/30) are tuned at 1:100; A1800 is 1:200 (every length is half). | door arcs fit to half-size; fragments filtered by 1:100 sizes. |
| 3 | Stroke semantics differ per sheet: Student clutter = 0.12 pt, walls 0.24-0.96; A1800 clutter = 0.21 pt, structure 0.43-1.91. A fixed 0.2 threshold keeps all of A1800's clutter. | stroke-length histogram per weight (`calibrate.width_classes`). |
| 4 | Wall-end extension was axis-aligned only; A1800 is made of diagonal and curved walls. | classrooms stayed connected to the concourse (one 3 536 m2 region before splitting). |
| 5 | Trees, grid bubbles and columns (hundreds of circles) became rooms or "bathrooms" through the round-fixture rule (9 false bathrooms). | v4 output: 73 rooms, 62 unknown, 9 bathroom. |
| 6 | Path extraction (per-point `pymupdf.Point`) cost ~30 s of a 51 s run. | stage timing. |
| 7 | Keyword matching on glued text: "Science Showcase" matched `wc` (toilet), "Product Design" matched `duct` (shaft). | found in the v5 A1800 output (a 3 583 m2 region typed `shaft_or_riser`). |

## 2. Changes (generic, no per-PDF constants; sheet-specific facts live in `config/profiles/*.yaml`)
| change | where |
|---|---|
| Path extraction through `get_cdrawings` and vectorised assembly: **x11.8 (Student) / x6.3 (A1800) faster**, output equal to the old extractor (max coordinate difference < 6.1e-5 pt; tested on a synthetic PDF with lines, curves, rect, quad, and with page rotation). Old extractor kept as `extract_paths_reference`. | `vector/extract_pdf_paths.py` |
| **Structural stroke threshold = 1.5 x the sheet's dominant line weight** (the weight that carries the most ink is clutter). Student: 0.12 -> 0.18 pt (same wall selection as the old 0.2); A1800: 0.21 -> 0.315 pt. | `vector/calibrate.structural_stroke` |
| **Automatic plan footprint / ROI** from heavy strokes (>= 3.3 x dominant): long straight lines (frame, site boundary, title block) are dropped, empty outlines (pond) are dropped, strokes are closed at 3 m, clusters with enough paths/ink are kept, thin spurs and necks (< 10 m) are cut. On A1800 it hugs the building (see `../evaluation_v5/A1800_v5_overlay_FULLPAGE.png`). | `vector/calibrate.plan_footprint` |
| **Printed-scale factor** `f = 100 / scale_ratio` applied to every pt parameter (door arcs, channel closing, inscribed radius, area, extension lengths, column sizes, classification areas). Test: the same 8 m x 6 m rooms are found at 1:100 and 1:200. The ratio itself cannot be read from these PDFs (the sheet text is outlined: 0 characters) and door-arc radii are ambiguous on Student (peaks at 12, 15 and 26 pt), so it is a **profile / config value**, not guessed. | `pipeline.py`, `classify.py` |
| **Direction-agnostic wall-end extension** (skeleton ends run on along their own direction up to 3 m, `vector.gap_closing`). Test with a 45-degree wall. | `pipeline.py` + `raster/segmentation.close_wall_gaps` |
| **Open-plan split**: a region > 150 m2 holding >= 2 room labels is split by watershed from the label seeds; a room reaches at most 9 m from its label, the remainder is the open space (concourse). Flagged `split_by_label_watershed` + review. | `pipeline.py` (`instances.open_plan_split`) |
| **Circles as shapes**: closed strokes that fit a circle (trees, grid bubbles, columns, tables) are `circle` elements with radius in pt and m; unlabelled round regions are no longer rooms. | `vector/elements.circle_elements` |
| School vocabulary (`classroom_or_lab`, `sports_or_assembly_hall`, `reception_or_entrance`); short keywords (<= 3 letters) match whole words only; `duct` removed. | `config/classes.yaml`, `classify.match_keywords` |
| Sheet profiles + `--profile` / `run.bat ... [profile]`: `student_accommodation.yaml` (ROI, footprint, exemplars, 1:100, old extension, no split) and `a1800_school.yaml` (only `scale_ratio: 200`). | `config/profiles/`, `cli.py`, `run.bat` |
| Dashed paths are not walls when no ROI is given; calibration values are written to `rooms.json -> image.calibration`. | `pipeline.py` |

## 3. Results
### 3.1 No regression
* **Student Accommodation** (with its profile) vs the v4 deliverable: 109 -> 109 rooms, 109/109 same type, **min IoU 1.0, max vertex displacement 0.0 pt**; elements identical (door 105, window 62, column 36, wall 54) plus 2 new `circle` elements.
* **floor_plan.pdf / floor_plan2.pdf** (raster mode, code unchanged): `expected.png` metrics v5 = mean 5.32 px, median 2.87, recall@8px 0.84, lines >= 80 % @8px 9/13, X-marks 0.0 coverage - identical to v4 (`evaluation_v5/expected_png_floor_plan_v5.json`).

### 3.2 A1800 (no ground truth exists; proxies, not accuracy; `evaluation_v5/a1800_proxies.json`, `python -m src.evaluation.vector_proxies`)
| proxy | v4 | v5 without open-plan split | v5 |
|---|---|---|---|
| rooms | 73 | 236 | 259 |
| the same 38 printed room names found inside a room | 8 | 37 | 35 |
| named labels sharing a room with another room's name (label isolation; 1.0 = none) | 1.0 (but only 8 labels in play) | **0.514** (one region holds 17 labels) | 1.0 - *by construction*, the splitter uses these labels |
| rooms > 150 m2 | 0 | 4 (max 3 536 m2) | 5 (max 1 898 m2 = the concourse) |
| room-boundary length within 2 pt of a structural stroke | 0.426 | 0.958 | 0.926 |
| `unknown` share of rooms | 0.849 | 0.627 | 0.564 (14 % of the area; median 2.6 m2: cupboards, risers, voids) |
| false bathrooms on trees / circle rooms | 9 / 0 | 13 bathrooms, 3 circle-shaped rooms, 260 `circle` elements | same |
Honest reading: v5 finds the whole building and separates the labelled rooms, but the label-isolation of the final result is not independent evidence (see "v5 without split": 0.514). The split lines are watershed cuts, not drawn walls. The split loses 2 of the 37 labelled rooms (reach limit).
Visual inspection (`evaluation_v5/A1800_v5_overlay_FULLPAGE.png`, `..._v4_vs_v5_side_by_side_FULLPAGE.png`): classrooms, labs, studios and lecture theatres are separate polygons; the concourse is one open-space polygon; pond, key plan and title block are no longer reported.

### 3.3 Full page, coordinates
Overlay, mask and reference render are the full page (A1800: 4737 x 3351 px at 1.5 px/pt = 3158 x 2234 pt, no crop or stretch). All 259 room polygons are valid; `polygon` (px) = 1.5 x `polygon_pt` within 0.055 px. **Tiling:** the wall raster of the plan region (about 9 400 x 4 700 px at 4 px/pt, 44 MP) is built in one piece (peak RAM 3.3 GB with CUDA, 2.2 GB CPU); only the OCR is tiled (1600 px tiles, 200 px overlap, mapped back to page points; unchanged). A tiled/bit-exact raster path was **not** implemented because it was not needed at this size - an A0 sheet at 2x the density would need it.

### 3.4 Speed and GPU (RTX 2050 4 GB, torch 2.11+cu128, onnxruntime-gpu 1.23.2; laptop on battery, so interleaved: `evaluation_v5/benchmark_a1800_cpu_vs_gpu.json`)
| A1800 | CPU-only env | GPU env, CUDA |
|---|---|---|
| pipeline (median of 2) | 75.2 s | **57.9 s (x1.30)** |
| OCR stage | 27.9 s | **11.0 s (x2.54)** |
| path extraction + arcs | 16.1 s | 15.9 s (CPU work) |
| wall mask / gap closing / labelling | 12.5 s | 12.2 s (CPU work) |
Sampled run: peak VRAM 1 688 MB (+1 267 MB over baseline), peak utilisation 100 %, mean 12.8 %, 26 of 153 samples above 10 %. Results CPU vs CUDA: identical rooms, types and polygons; one OCR box (a spurious CJK glyph) differs. Only OCR (ONNX CUDA) and the symbol template matcher use the GPU; the rest is OpenCV/NumPy on the CPU. Moving the door-arc fit or the skeleton to the GPU was not attempted.
Run time of the final deliverable runs: A1800 69 s (v4: 51 s but on a 6x smaller window of the sheet and with 8/38 labels), Student 16.5 s (v4 ~24 s), floor_plan 5.6 s, floor_plan2 4.5 s.

## 4. Tried / decided against
* Reading the scale from the PDF (no text layer) or inferring it from door-arc radii (ambiguous on Student) - kept as an explicit profile value.
* Long thin lines as glazed screens: no reliable signature; the corridor-side openings of the classrooms are not drawn at all, hence the label-based split.
* Footprint from all strokes >= 1.5 x dominant: includes trees, site lines and the key plan; heavy-stroke footprint (3.3 x) is used instead.

## 5. Remaining limitations
* No ground truth for A1800/Student beyond printed labels; hand-annotate ~15 rooms to get IoU.
* Open-plan split boundaries are watershed/reach cuts (curved), flagged for review; the 1 898 m2 concourse and 4 other regions > 150 m2 remain single polygons.
* 146 `unknown` rooms (14 % of the area), 37 of them room-sized but unlabelled; furniture-weight partitions are not modelled.
* The printed scale must be supplied (profile) for sheets that are not 1:100; wrong ratio -> wrong m2 and wrong pt thresholds.
* Student needs its profile (symbol exemplars cannot be inferred); without it the generic mode still runs but does not find pods/windows by template.
* Footprint heuristic: the Student sheet's "NOTES" stamp box is included when the footprint is automatic (the profile fixes it); not tested on other sheets.
* A GPU speed-up exists only for OCR; for A1800 the non-OCR (CPU) stages are about 80 % of the GPU run (47 of 58 s).

## 6. Reproduce
```powershell
cd floor-plan-detection
.venv\Scripts\python.exe -m pytest tests -q                                   # 68 tests
..\run.bat "..\A1800 Ground Flr.pdf" "..\output_v5\A1800" auto a1800_school  # 4th argument = sheet profile
..\run.bat "..\Student Accomodation.pdf" "..\output_v5\Student" auto student_accommodation
.venv\Scripts\python.exe -m src.evaluation.vector_regression outputs\Student_Accomodation ..\output_v5\Student_Accomodation
.venv\Scripts\python.exe -m src.evaluation.vector_proxies "..\A1800 Ground Flr.pdf" ..\output_v5\A1800_Ground_Flr --scale 200 --min-stroke 0.315
.venv\Scripts\python.exe -m src.evaluation.benchmark_device "..\A1800 Ground Flr.pdf" bench --rounds 2 --profile config\profiles\a1800_school.yaml
```
