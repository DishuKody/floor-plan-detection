# v7: A1800 against the new hand-annotated reference - what was measured, and why the match is NOT yet satisfactory

**Status: no accuracy improvement was delivered in this round.** The detection output is identical to v6. One experimental feature was built and measured, and left switched off. 78 tests pass.

## 1. What the annotation says
`A1800 Ground Flr_expected.png` (1482 x 702; registered to the page render by SIFT + RANSAC, scale 2.63, 82 inliers, RMS 0.48 px) shows red lines on the **left half** of the plan. They are not walls: they are the **room outlines**, drawn along
* the outer wall and the partitions that already exist (these are easy), and
* the **open frontage of every classroom wing** - the line that joins the ends of consecutive partitions along the corridor / concourse edge, often curved or angled (Science Labs, SS+H, Maths, AS 01-06, the fan-shaped showcases), plus the whole outline of the lecture theatre (only its left half).
So the target is **a closed polygon per room**, not "the walls".

## 2. How it was measured (new: `src/evaluation/annotated_regions.py`, `run_v7_evaluation.py`)
Expected rooms = the enclosed regions of (structural walls rasterised straight from the PDF + the red lines, closed by door chords); 49 of them. Each is scored by the IoU with the **best single detected room** (a room merged into a neighbour, or split into pieces, both score low). This is stricter than the earlier line-distance recall, which a dense polygon mesh passes without drawing any correct room.

## 3. Results (v6 = v7, unchanged)
| metric | value |
|---|---|
| expected rooms with IoU >= 0.5 / 0.7 / 0.85 | **61 % / 45 % / 12 %** |
| mean / median IoU | **0.52 / 0.66** |
| annotated line pixels within 10 px of any detected boundary | 90 % (generous: boundaries are everywhere) |
| line pixels whose two sides lie in different rooms (both in rooms) | 0.653 (0.466) |
Visual check (`evaluation_v7/A1800_side_by_side_annotation_detected_comparison.png`, panel 3: green = matched, red = missed, blue = detected but not annotated): the outer envelope, the long partitions and the lecture-theatre outline are matched; the misses are the **corridor-side frontage** segments (short red pieces between wing partitions and across the fan-shaped rooms).

## 4. Root causes found (this round)
1. **The wing rooms are not closed rooms in the geometry.** Their corridor side has no wall, so each wing's free space is connected to the concourse. Example: AS 02 (45 m2, expected) is 93 % inside one 1,394 m2 "communal" polygon; the detected room that overlaps it is a 3.6 m2 fragment (IoU 0.06). The same happens for Science Lab 04, AS 03, AS 04, Cloaks, the SS+H wing: IoU 0.01-0.1.
2. **The frontage cannot be read from wall tips here.** Partition ends are fused into returns / blobs / door frames, so skeleton tips are few (476 on the whole sheet) and are not on the frontage line.
3. **False boundaries remain**: a long horizontal line along the A-A section axis through the concourse and the lecture theatre, and many 'extra' boundaries (furniture, stairs, tiers) - visible in panel 2/3.
4. The earlier open-plan split cuts at 9 m around a label, so it produces disks, not frontage lines.

## 5. What was tried (and measured)
* **Mouth closure** (`src/vector/mouths.py`, config `vector.mouth_closure`): link two wall tips when the link turns an *open* region into a bounded, compact, room-sized pocket (never cuts a closed room; shortest links first; direction and support-by-a-third-tip rules). Without rules: 63 links, many long wrong diagonals. With rules: 15 links, mean IoU **0.517 -> 0.554**, share >= 0.5 0.61 -> 0.67, but share >= 0.85 0.12 -> 0.08 and some links still wrong (see `evaluation_v6_runs/a1800_m3`). **Judged not good enough: switched off** (`enabled: false`); code and debug output (`debug.mouths`) kept for the next attempt.
* Expected-room evaluation itself (new, tested).

## 6. Regressions (default config, `output_v7/`)
Student: 109/109 rooms, same types, IoU 1.0, 0.0 pt; floor_plan / floor_plan2: polygons and types identical to v6; `expected.png`: mean 5.32 px, recall@8 0.84, 9/13 lines >= 80 %. Full-page A1800 overlay 4737 x 3351 (page render size), no crop.

## 7. What I would do next (not done)
* Reconstruct the frontage as a **smooth curve through the partition ends of one wing** (ordered tips along the wing, fitted with a spline, rooms = cells between neighbouring partitions) instead of pairwise links; needs reliable partition-end detection (end caps of fused blobs).
* Remove the section-axis line (a long thick horizontal not belonging to any room).
* Use room labels as seeds *within the pocket between two partitions*, not as 9 m disks.
Until then the corridor-side boundaries of the classroom wings are the main unresolved inaccuracy.
