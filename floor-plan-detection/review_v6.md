# v6: improving A1800 against the hand-annotated reference, without touching the other plans

Reference: `A1800 Ground Flr_expected.png` (1522 x 886) - a screenshot of an earlier overlay (a ~0.35x crop of the page) with room boundaries drawn in red on the **left half** of the plan (partitions and the open corridor-side edges of the classrooms), and a red X over the eco pond ("not a shape").
All numbers were produced by `python -m src.evaluation.run_v6_evaluation` (`../evaluation_v6/summary.json`); v5 is `../output_v5/`, v6 `../output_v6/`. Tests: **76 passed** (68 + 8 new in `tests/test_vector_v6.py`).

## 1. How the reference is read (new, `src/evaluation/annotated_reference.py`)
* The screenshot is registered to the page render with SIFT + RANSAC (affine; scale 2.857, RMS 0.56 screenshot px, 102 inliers); the old tool assumed a full-page screenshot, so a `scale_hint` was added. The file is never modified.
* The red strokes join into networks: they are classified by topology (a cross = "not a shape"; everything else = boundary lines); review rings and dots are ignored.
* **Distance recall alone is misleading here** (v5 already scores 0.88: a dense mesh of polygons is near every line). The metric that matches what the lines *mean* is **separation**: the share of line pixels whose two sides (+-10 overlay px) lie in different detected rooms (`separated_both_in_rooms`: both sides in rooms). Both are reported.
* The lines are sparse (partial annotation), so **precision cannot be measured**; over-segmentation is tracked with reference-free proxies instead.

## 2. Root causes found (evidence in `evaluation_v6_runs/`)
| # | cause | evidence |
|---|---|---|
| 1 | Of the line pixels v5 failed to separate, 76 % had a structural stroke under them: the walls are drawn, but both sides ended in the same room: **48 % of all annotated pixels had the same room on both sides** (leaks), 41 % separated two rooms. | stroke-class lookup under each failing pixel |
| 2 | The classroom partitions meet the corridor in a broken line of short thick stubs with 5-10 m openings; **glazed screens are drawn in the clutter weight** (0.21 pt, same as furniture), so they were thresholded away. Long straight clutter-weight strokes inside the building trace these screens (visual check). | `thin.png` overlay of such strokes vs the annotation |
| 3 | **Dash-dot sight / axis / grid lines at any angle** (the existing filter handled only 0.24 pt axis-aligned lines of the Student sheet) acted as false walls: long straight false boundaries across the concourse and 30+ false free ends per axis. | 425 -> 244 wall-end extensions once removed |
| 4 | The wall thickness used by the wall-end closure came from a high percentile over poche blobs (10.8 pt instead of ~1.2 pt): short stubs were pruned as spurs, so their ends never closed. | thickness estimate vs medial axis |
| 5 | **A real bug**: `region_polygon` lost any region whose contour pinches at a pixel joint (`make_valid` returns a GeometryCollection that *contains* a MultiPolygon; only top-level polygons were searched). The largest regions (concourse + wings, 1000+ m2) were silently dropped. | probe diagnostics: "polygon conversion failed"; 3 regions dropped per run in v5 |
| 6 | A region was kept or dropped by a single representative point against the footprint. | probes |

## 3. Changes (generic; sheet specifics stay in profiles)
| change | where |
|---|---|
| `find_dash_chains`: dash / dash-dot lines at any angle and weight (>= 6 collinear segments, visible gaps, **regular** gaps - median tolerance, or two alternating gaps). Window / door openings have irregular gaps and are not matched (unit-tested). | `vector/walls.py` |
| `find_long_thin_lines`: long (>= 4 m) straight clutter-weight strokes inside the building = glazed screens / partitions; regular parallel series (hatch, treads) dropped; closed back-and-forth paths ignored. Added **after** the wall-end extension and only for room separation (wall *elements* keep the structural mask). | `vector/walls.py`, `pipeline.py` |
| Wall thickness from the medial axis; end walks long enough to see junctions; a minimum wall-run length of 2 m (`min_end_branch_m`); wall ends / corners are joined up to 12 m (`pair_gap_m`), any wall up to 3 m. | `pipeline.py`, `raster/segmentation.close_wall_gaps(endpoint_kw=...)` |
| `largest_polygon` flattens nested geometry collections (bug 5); the footprint test clips regions that lie mostly inside an automatic footprint. | `postprocessing/polygons.py`, `pipeline.py` |
| Diagnostics: drop reasons per region and `debug.probe_points_pt` (why is the region at this point kept or dropped, per stage). | `pipeline.py` |
| Evaluation: annotated-reference tool, separation metric, `run_v6_evaluation`. | `src/evaluation/` |

## 4. Results (A1800, annotated left half)
| metric | v5 | v6 |
|---|---|---|
| line pixels separating two rooms (both sides in a room) | 0.408 | **0.436** |
| line pixels with different rooms on the two sides (any side) | 0.507 | **0.588** |
| distance recall @ 8 px / mean distance | 0.883 / 4.54 px | **0.905 / 4.25 px** |
| pond X: largest room covering it | 0.0 | 0.0 |
| rooms > 150 m2 (merged concourse / wings) | 5 | **1** |
| printed room names inside a room (of 38) | 35 | 35 |
| rooms / `unknown` share | 259 / 0.564 | 380 / 0.632 |
| boundary length within 2 pt of a structural stroke | 0.926 | 0.807 (boundaries now also run on clutter-weight screens, which this proxy does not count) |
Sensitivity to the sampling offset (the lines are hand-drawn, so the offset is a free choice): any-side separation v5 -> v6 is 0.575 -> 0.600 (6 px), 0.507 -> 0.588 (10 px), 0.505 -> 0.575 (14 px); both-sides-in-rooms is 0.261 -> **0.224 (6 px, i.e. worse)**, 0.408 -> 0.436 (10 px), 0.412 -> 0.471 (14 px). At 6 px the samples often fall inside the wall band, so the improvement is clear only for offsets above the hand-drawing error; the gain in the headline 10 px figure is modest (+0.03). Full-page images: `../evaluation_v6/A1800_v5_vs_v6_side_by_side_FULLPAGE.png`, `A1800_expected_vs_detected_v5_vs_v6_FULLPAGE.png` (red = expected, blue = detected), per-version `a1800_vs_annotation_v5|v6/`.

## 5. Regressions checked
* **Student Accommodation** (profile): 109/109 rooms, same type, **min IoU 1.0, 0.0 pt displacement**, elements identical (+2 `circle`). The dash rule is switched off in its profile because it matched real window-glazing / door-sill runs (1.3 % change of two pods); without that switch the result was not bit-identical.
* **floor_plan.pdf / floor_plan2.pdf** (raster mode): polygons, types and elements identical to v5; `expected.png`: mean 5.32 px, recall@8 0.84, 9/13 lines >= 80 % - unchanged.
* Earlier attempts that were **rejected after measuring**: a free-end "alcove closure" (links cut through rooms; whole regions disappeared), a row-of-ends chain closure (links only on the site perimeter), directional line-closing (over-fills wedges and curved walls), thin lines added before the wall-end extension (changed the skeleton and re-merged rooms), including closed back-and-forth paths as screens (lowered agreement and broke Student parity).

## 6. Unresolved
* **The corridor-side open edges are still mostly unmatched** (the annotation's main message): where a classroom opens onto the corridor over 5-10 m with no screen drawn at all, nothing in the geometry marks the edge. Only 43.6 % of annotated pixels separate two rooms; per-line results range from 0.07 (the fan-shaped "Coms 01 Showcase", bounded only by furniture-weight arcs) to 1.0.
* More rooms and more `unknown` ones (cupboards, risers, screens' pockets): precision is not measurable here; judge by the overlay.
* Open-plan splits across the concourse remain label-based watershed cuts (curved), flagged for review.
* The dash rule is sheet-sensitive (see Student); the long-thin-line rule assumes screens use the dominant clutter weight.
* Only the left half of A1800 is annotated; the right half (sports hall) has no reference.
* A1800 now takes ~70 s (the thin-line and end analysis add time; v5 ~58 s on GPU).
