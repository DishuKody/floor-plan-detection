# v8: independent area segmentation of A1800 (areas as defined by the red annotation)

Reference: `A1800 Ground Flr_expected.png`. Run: `../output_v8/`; evaluation + images: `../evaluation_v8/` (`summary.json`). Tests: **80 passed**. The PDF used is `A1800 Ground Flr.pdf` (the path in the request, `A1800 Ground Flr/pdf`, does not exist; `A1800 Ground Flr(1).pdf` was not found on disk).

## 1. What was wrong
Every classroom wing is drawn as outer wall + partitions with **nothing drawn on the corridor side**. The free space of each wing is therefore connected to the concourse, and one polygon covered many of the rooms you drew (e.g. AS 02, 45 m2, sat inside a 1,394 m2 area). Earlier versions split such a region only from keyword-matched labels, clipped each part to a 9 m disk, and the 'rest' became one concourse polygon.

## 2. What changed (generic, in `pipeline.py` open-plan split; no coordinates, no PDF-specific rules)
1. **Seeds from room-name text** (any text with >= 2 letters inside the region, grouped by proximity) instead of a keyword list; this includes names such as "AS02" that no keyword matched.
2. **Core seeds in unnamed wide space**: distance-transform maxima (half-width >= 1.6 m, >= 6 m from every label) - so corridors / the concourse get their own seeds and are not absorbed by the neighbouring room.
3. **A region is split when it has a label and an open core** (before: >= 2 labels), from 100 m2 (before 150); the 9 m disk clip is removed.
4. **Thin-strip trimming**: a room is "fat", the corridor strip its watershed basin spills into is thin. Each room part is opened (mitre joins, sharp corners kept) by 0.6 x its own inscribed radius; what is trimmed goes to the open-space polygons.
5. **Output you asked for**: `areas_overlay.png` (full page, drawing visible, every area with its **unique ID** and a **distinct colour**; neighbouring areas never share a colour, 60-colour palette, least-used first), `areas.json` (id, colour, type, area, polygon in pt and px, holes, centroid, label point, shape, flags). Also copied as `<name>_areas.png/.json`.
Parameters were chosen by sweeping on this one reference (reach, trim 0.4 / 0.6 / 0.8, area threshold, letters), so some fit to A1800 is possible; the Student sheet keeps the split off in its profile.

## 3. Results (49 expected rooms = enclosed regions of the PDF's structural walls + your red lines)
| metric | v6 | v8 |
|---|---|---|
| rooms with IoU >= 0.5 | 61 % | **76 %** |
| rooms with IoU >= 0.7 | 45 % | **57 %** |
| rooms with IoU >= 0.85 | 12 % | **29 %** |
| mean / median IoU | 0.52 / 0.66 | **0.65 / 0.73** |
| your red lines with different areas on the two sides | 0.65 | **0.79** (both sides inside areas: 0.47 -> 0.58) |
| classification of the 49: good / merged in a larger area / split-or-partial / inaccurate boundary | 22 / 17 / 6 / 4 | **28 / 11 / 3 / 7** |
| areas in the whole sheet | 380 | 405 |
Images: `A1800_annotation_vs_independent_areas.png` (your annotation above, our coloured IDs below, same frame), `A1800_areas_FULLPAGE.png` (4737 x 3351 = page render, uncropped), `A1800_side_by_side_annotation_detected_comparison.png` (green matched / red missed / blue extra), `A1800_room_iou_map_FULLPAGE.png` (green >= 0.7, orange 0.4-0.7, red < 0.4).

## 4. Still wrong (measured / visible)
* **11 of 49 expected rooms are still merged into a larger area**, e.g. several AS wing rooms and Science Lab 04; **7 have an inaccurate boundary** (the frontage is a watershed cut, not your line).
* **The lecture theatre** is not one outline: it is split into 4-8 areas (seating, stairs, side bays); the triangular showcase rooms are cut into several pieces.
* **The concourse** is cut into several large areas (IDs 0065, 0136, 0213, 0319 ...) by watershed lines that are not architectural; flagged `split_by_label_watershed`.
* **Over-segmentation outside the annotated half** (the sports hall and right wing have many tiny areas); not annotated, so unscored.
* The frontage lines are straight / curved cuts between distance-transform saddles; they follow your lines closely where a neck exists and are off by up to ~2 m elsewhere.

## 5. Regressions
Student (profile): 109/109, same types, **IoU 1.0, 0.0 pt**; floor_plan / floor_plan2: polygons and types identical to v6; `expected.png`: mean 5.32 px, recall@8 0.84, 9/13 lines >= 80 % (unchanged).
