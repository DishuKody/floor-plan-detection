# Review of v1 output and the v2 changes

Reference: `Student Accomodation.pdf`. v1 = the previous run (`versions/v1/`, code + config + outputs, reproduces bit-for-bit).
v2 = current (`outputs/`, `src/`, `config/`). All numbers below are recomputed from the output files by
`python -m src.evaluation.compare_versions versions/v1/outputs outputs outputs/comparison` -> `outputs/comparison/comparison.md|json`
and side-by-side panels `outputs/comparison/*.png`.

## 1. What I checked in v1 (against the drawing)

| Check | Finding in v1 |
|---|---|
| Adjacent rooms merged | Bedrooms: none (0 merged of 39 labelled units). **Circulation: yes** - one 72.1 m2 region fused the Core 03 lobby, the Riser room, the AOV room and both upper-row corridors. |
| Bathrooms / small rooms missed | **AOV not a room at all**; Riser swallowed; Stair 03 only its 6.0 m2 landing; Tutor Flat only 3.4 m2. Annex units (3) outside the configured footprint (known, unchanged). OCR never found "Clnrs", so that space is unchecked. |
| Wrong classification | The 72.1 m2 region was typed `shaft_or_riser` (the word "Riser" won over region size); the real lobby was therefore mis-typed. 9 `unknown` rooms. |
| Boundaries through walls | **Not found**: no v1 polygon contains more than 0.6 % wall ink; boundaries sit a median 0.5 pt from drawn wall faces. The real boundary weakness is the bathroom boxes (their edges lie in open space) and the strips they leave behind. |
| Corridor classification | Corridors themselves OK, but the upper-row corridors were absorbed by the lobby region (same 72 m2 error). |
| Invalid / overlapping polygons | 0 invalid, 0 overlapping pairs. |
| Coordinate mismatch | None: boundary-to-ink distance 0.83 px (mean of per-room medians), `px = pt * 1.5` exact to 0.055 px. |

## 2. The three most significant errors, root causes, fixes

### E1 - Circulation core fused and mis-typed (72.1 m2 `shaft_or_riser`)
Root causes (each found by looking at the actual paths, several of my first guesses were wrong - see section 4):
1. **Door arcs split into fragments.** The riser's swing is stored as two paths of 37 deg and 39 deg sweep; my detector required 60-120 deg per path, so the door was never detected and the opening stayed open.
2. **Wrong radius chosen.** Even for detected arcs the closing chord was drawn along the *open leaf* (into the room) because the evidence test only looked behind the hinge and both radii touch a jamb cap (tie -> first).
3. **Chord shorter than the framed opening** (swing radius 26 pt vs ~35 pt opening) leaving a ~9 pt gap (AOV).
4. **Heading text treated as wall.** "Core 03" (14 pt glyphs) passed the "long black fill = wall poche" test, punching holes in the lobby and pushing its label out of it.
5. Classification let a keyword ("Riser") override an implausible 72 m2 area.

Fixes: merge arc fragments by common centre/radius before the sweep test (`vector/doors.py`); choose the radius that has wall behind the hinge **and** beyond its end, and extend the chord <= 14 pt to the far jamb (`candidate_regions.close_doors`); accept thin 0.12 pt arcs only with wall evidence; restrict black wall fills to 3-5-point polygons (glyphs have >= 6) (`walls.py`).

### E2 - Core-cluster rooms incomplete: Stair 6.0 m2, AOV missing, Tutor Flat 3.4 m2
Root cause: symbol linework inside enclosures subdivides the interior - stair-tread lines leave only the landing above the 7 pt minimum-radius filter; the AOV's diagonal "X" cuts it into slivers below that filter.
Fixes: remove equally spaced parallel tread series (8 segments) from the wall mask (`vector/symbols.py`); a second pass for named structural spaces (labels stair/lift/riser/AOV) that re-segments from thick structural walls + door chords only and is accepted only when closed, 1-60 m2 and not swallowing a bigger room (`postprocessing/structural.py`). Note: this pass is label-seeded.

### E3 - En-suite pods: template box misses part of the pod and leaves stray rooms
Root cause: the 5-bed pod is taller than my exemplar box, so a 1.4 m2 strip of each pod stayed behind as a separate `unknown` room (3 of them touching a bedroom). I first assumed they belonged to the bedroom; the shared-boundary numbers showed they share a 46 pt side with the pod and are walled off from the bedroom.
Fix: attach unlabelled sub-4 m2 pockets to the bedroom/bathroom with the longest shared boundary, as `extra_polygons` with a flag (`adoption` config). 7 pockets adopted.
Not fixed: the pod outline is still the matched symbol's bounding box (+ adopted pieces).

## 3. Measured comparison (v1 -> v2)

| metric | v1 | v2 | verdict |
|---|---|---|---|
| Named spaces correct (stair, riser, 2 lifts, AOV, store, Core 03 lobby, tutor flat) | **3/8** | **7/8** | improved |
| Largest room | 72.1 m2 `shaft_or_riser` | 37.6 m2 `communal_or_shared_space` (lobby) | improved |
| Riser / AOV / Stair | merged / merged / 6.0 m2 | 2.6 m2 own room / 2.7 m2 own room / 10.5 m2 | improved |
| Tutor Flat main room | 3.4 m2 | 13.9 m2 (my plausible range 15-80 still fails) | partly improved, still FAIL |
| `unknown` rooms | 9 | 4 | improved |
| Unknown pockets touching a bedroom | 3 | 0 | improved |
| Bedroom units isolated+correct (OCR unit-ID proxy) | 37/39 | 37/39 | **no change** (the 2 misses are the out-of-footprint annex) |
| Kitchen units | 6/7 | 6/7 | no change |
| B703 bedroom area CV (24 identical units) | 0.007 | 0.007 | no change, already consistent |
| Invalid polygons / overlaps | 0 / 0 | 0 / 0 | no change |
| Boundary > 3 pt from a wall (mean fraction) | 0.123 | 0.123 | no change |
| Max wall-ink inside a polygon | 0.006 | 0.097 | explained: room-0049 is the AOV, whose interior contains its own X symbol; not a boundary error |
| Coordinates (boundary-to-ink px / max abs px-pt*s) | 0.83 / 0.055 | 0.80 / 0.055 | no mismatch |
| Doors / windows / columns / wall polygons | 92 / 14 / 36 / 3 | 105 / 14 / 36 / 54 | doors +13 (not individually verified); walls: polygon repair (v1 showed 3 coarse polygons); windows unchanged |
| Runtime / peak RAM | 39.4 s / 997 MB | 24.4 s / 998 MB | CPU only |

Caveats on these numbers:
* The named-space expectations (type + plausible area) are my own reading of the drawing, not ground truth; the AOV/stair result partly relies on the same OCR labels the check uses, so it is not fully independent.
* B503 bedroom area spread (CV 0.073 -> 0.077, 2 outliers) did **not** improve; it is also not a valid test because 07/5 (9.8 m2) and 07/4 (8.4 m2) are genuinely different layouts. Only B703 is a valid consistency test.
* Still no manual ground truth: IoU and boundary F-score are **not** established.

## 4. Things I tried that did not work (kept out of the pipeline)
* Directional morphological closing of thick walls (30 pt) to seal core doors: added 824,500 px (it fills *between* parallel walls) and collapsed the wall output. Removed.
* Accepting 0.12 pt arcs alone: door count 92 -> 115 but the 72 m2 region did not change. The real blocker was arc fragmentation + wrong radius (above).
* A crossing-diagonal ("X") symbol remover: also fired on unrelated pod-door diagonals and missed the AOV's X. Not enabled (`find_cross_symbols` exists in `vector/symbols.py` but is unused).
* Thick-wall-only segmentation for the stair: its east walls are thin lines, so it leaks; only the AOV is handled that way. The stair is fixed by tread removal.

## 5. Would a pretrained model, higher resolution or fine-tuning help?
* No pretrained model produced these errors - the pipeline has none; all three were deterministic geometry/post-processing faults, and **vector geometry extraction plus better post-processing fixed them** (7/8 vs 3/8).
* Higher-resolution / tiled inference is irrelevant for the vector path (lossless); it matters only for the raster plans, which this pipeline still rejects. Raster2Seq remains blocked (no nvcc/MSVC).
* Fine-tuning is not justified by these errors. It would only be considered for raster inputs, after real annotations give a baseline.

## 6. Remaining known problems (v2)
Tutor Flat main room 13.9 m2 vs my 15 m2 lower bound (the unit is split over two regions); 4 `unknown` rooms (three east-end lobbies, a 1.6 m2 piece under the Riser); the lobby polygon includes the cross-corridor strip between the cluster fire doors; windows incomplete (14, none on the south facade); bathroom polygons are symbol bounding boxes; annex outside the footprint; 48 of 107 rooms flagged `review_required`; no manual ground truth.
