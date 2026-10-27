# Floor-plan detection — analysis & results report

Reference drawing: `Student Accomodation.pdf` (Aston Student Villages, Block D, second floor plan, Core 3).
Everything below was measured on this machine; anything not measured is labelled as such.

## 1. Input analysis

| File | Pages | Content | Vector geometry? |
|---|---|---|---|
| `Student Accomodation.pdf` | 1 | 2384 x 1684 pt (A1), page rotation 270, printed scale 1:100 (title block) | **Yes**: 38,609 drawing objects (49,372 paths after splitting), 0 embedded images, 0 text-layer words (text is outlined glyphs) |
| `floor_plan.pdf` | 1 | A4 page holding one 736 x 941 grayscale raster; small residential plan, 3 bedrooms, baths, WC, kitchen | No (0 paths, no text layer) |
| `floor_plan2.pdf` | 1 | A4 page holding one 600 x 794 grayscale raster; 2-bed bungalow | No |

I treated `Student Accomodation.pdf` as *the* reference plan: it matches the brief (repeated bedroom clusters, small
en-suites, corridors, communal spaces, dense linework). The two raster files are small residential plans; the pipeline
refuses them with an explicit message (see §6).

Drawing characteristics (verified in the vector data and by zooming):
* Stroke width carries semantics: 0.12 pt furniture/sanitary symbols, 0.24 pt partition walls (double lines), 0.48 pt
  windows/door frames/door leaves/swing arcs, 0.72 pt kitchen units and some walls, >=0.96 pt and black fills for external
  wall/core poche, grey-filled triangles for columns.
* **Walls are not closed polygons.** Door openings are gaps, partitions are drawn up to ~0.8 m short of the wall they meet,
  door swings are stored as many short line segments (no Bezier curves, no dash attributes), grid lines are dash-dot
  chains of single 0.24 pt segments.
* Rooms: 7-bed clusters (4 rows x 7 bedrooms), 5-bed clusters (2 x 5), cluster kitchens, stair core, two lifts, riser, AOV,
  store, cleaners, tutor flat, corridors. Every bedroom carries a unique unit-ID label such as `02/D3/12/7`.
* The en-suite of each bedroom is a repeated CAD block (identical copies, mirrored/rotated), inside the bedroom, with
  its own partial wall bracket.

Likely failure modes (all confirmed during the work): merged bedroom pairs, door-gap leakage, wall-cavity and window-strip
slivers, text/furniture strokes breaking regions, corridor/lobby merging, en-suites swallowed by bedrooms.

## 2. Environment (inspected before installing)

* Windows 11 Home 10.0.26200; system Python 3.14.5 only (too new for most ML wheels) -> isolated venv with uv-managed
  CPython 3.11.16.
* RAM 15.2 GB. NVIDIA GeForce RTX 2050, 4 GB, driver 592.27 (reports CUDA 13.1), 273 MB used by other apps.
  AMD Radeon 740M iGPU: not used, not assumed CUDA-capable.
* **No `nvcc`, no MSVC / Visual Studio build tools.** PyTorch is *not installed* (not needed by the selected approach).

## 3. Approach selection

| Option | Verdict (this drawing, this hardware) |
|---|---|
| **Vector-first** (PyMuPDF + Shapely + OpenCV) | **Chosen.** The source has full vector geometry; resolution is lossless, no tiling/downscaling, CPU-only, deterministic. |
| Raster2Seq (MIT code, HF checkpoints `haopt/Raster2Seq`) | **Blocked, and a poor fit.** Tested on Linux/PyTorch 2.3.1/CUDA 11.8; needs compiled CUDA ops (deformable attention) and `diff_ras`; this machine has no nvcc/MSVC. Checkpoints are trained on residential datasets (Structured3D, CubiCasa5K, Raster2Graph) at ~512 px, which would erase 0.24 pt walls and 1 m^2 en-suites on an A1 sheet. Not installed, not run. |
| CubiCasa5K | Already tried by the user; not repeated. |
| SegFormer / Mask2Former / DeepFloorplan | Not run: no floor-plan checkpoints that fit this CAD style; would need fine-tuning data that does not exist yet. |
| SAM | Not run: no room semantics, adds heavy dependencies without evidence of benefit. |
| OCR (RapidOCR / ONNX Runtime, CPU, local) | **Used** as a supporting signal (labels, unit IDs). Nothing is sent off-machine. |
| Template matching of the repeated en-suite block | **Used** as the fixture detector. Needs one exemplar box per symbol (user-in-the-loop input, in `config/inference.yaml`). |

No GPU is used: the vector path is CPU-only, so there is no VRAM figure (the 4 GB budget is irrelevant to the method).

## 4. Pipeline (what actually runs)

1. Extract all paths (curves flattened, rotation applied) -> displayed page points.
2. Remove dash-dot grid lines (chains of collinear 0.24 pt segments); detect door-swing arcs by circle fitting
   (92 found; radius 12-40 pt, 60-120 deg sweep) and remove them from the wall mask.
3. Rasterise strokes >= 0.2 pt + grey and long black fills at 4 px/pt; extend dangling partitions to the wall they meet
   (including through door-frame gaps); close each door opening with a hinge-to-jamb chord.
4. Fill double-line wall channels, label the free space, keep regions with inscribed radius >= 7 pt inside the building
   footprint, drop fragments < 1 m^2 (listed in `outputs/debug/discarded_fragments.json`).
5. OCR on the plan area (tiled, 3x upscale); template-match the en-suite symbol (mirrored/rotated) and cut each match out of
   its bedroom as a separate `bathroom` room.
6. Rule-based classification with explicit evidence (OCR keywords > fixtures > geometry), polygons with holes,
   `review_required` flags. Optional label-guided watershed split (off by default, see §5).
7. Elements: doors (hinge, leaf radius, swing, opening), columns, windows (partial), wall polygons.

## 5. Results (measured)

Run: `python -m src.cli "../Student Accomodation.pdf"` -> 39.4 s wall time, 997 MB peak RAM, CPU only.

Output: 106 rooms: 38 bedroom, 39 bathroom, 6 kitchen, 7 corridor/hallway, 2 lift, 1 stair, 1 riser/shaft, 1 storage,
2 other (tutor flat parts), 9 unknown. 92 doors, 36 columns, 14 windows, 3 wall polygons (with holes).
All polygons valid (0 invalid, 0 overlapping pairs > 2 %).

### Proxy accuracy (NOT manual ground truth)
There is no hand-annotated ground truth. The only independent signal is the drawing's own unit-ID labels
(`02/D3/12/7` ...): OCR read **46** of them (39 bedroom units, 7 kitchen units). For each label I checked which predicted
room contains it, whether that room contains any other unit label (= merged) and whether its type is right.

| | units | isolated & correct | merged | wrong class | missed |
|---|---|---|---|---|---|
| bedrooms | 39 | **37** | 0 | 0 | 2 |
| kitchens | 7 | **6** | 0 | 0 | 1 |

* The 3 misses are the "Core 2" annex at the far right (units 02/D2/06/4, /5, /K). It is a fragment of another drawing and lies
  outside the configured building footprint, so it is deliberately not processed.
* Before the en-suite cut-outs were added the same metric gave 31/39 bedrooms isolated (3 merged pairs); the cut removes the
  narrow neck through which the pods joined neighbouring bedrooms. The label-guided watershed split reached the same count
  but is not independent of this metric, so it is disabled by default (`instances.split_by_labels`).
* This proxy covers **only labelled bedrooms/kitchens**. It says nothing about boundary accuracy, bathrooms, corridors or
  communal spaces. **Per-class IoU, boundary F-score and bathroom/corridor precision-recall have not been established.**
* The en-suite check is not measured at all by the proxy: the 39 pod matches were verified visually only (28 in the four
  7-bed rows, 10 in the 5-bed wing, 1 in the tutor flat — consistent with the drawing).

### Visual inspection findings (known defects)
* Corridors: the 7-bed corridors and the Core 03 lobby are merged into one large `corridor_or_hallway` region per row/area.
* Bathroom polygons are the *bounding box of the matched symbol* intersected with the room, not the true pod walls
  (flag `boundary_is_symbol_bounding_box`).
* Windows: only 14 found (north facade mostly; none on the south facade) — the detector is incomplete.
* 9 `unknown` regions (east-end lobbies, small pockets next to pods in the 5-bed wing, store annexes); tutor flat is split
  over two regions.
* Door-opening chord selection is heuristic (92 doors; 8 had no wall evidence for either radius).
* `confidence` is always `null`; `heuristic_score` is a rule strength (or a template correlation), never a probability.
  52 of 106 rooms are flagged `review_required`.
* Areas in m^2 assume the printed 1:100 scale and are approximate.

## 6. Raster-only inputs
`floor_plan.pdf` / `floor_plan2.pdf` have 0 vector paths; the CLI stops with a clear error. Handling them needs a raster
model. Raster2Seq cannot be built here (no compiler toolchain); this is the exact remaining blocker.

## 7. Recommended next steps (smallest experiments first)
1. Hand-annotate ~15 rooms (bedroom, en-suite, corridor, lobby, stair, store) on this sheet to get real IoU / boundary F1.
2. Split the merged corridor/lobby regions with the same neck-based approach (door chords across corridor openings) and
   label them from "Core"/"Cluster" text only where unambiguous.
3. Windows: detect parallel multi-line glazing symbols by template matching (same mechanism as the en-suite pod).
4. For raster plans: install Visual Studio Build Tools + CUDA 11.8 toolkit (or WSL2/Linux) to try Raster2Seq, or use a
   segmentation model fine-tuned on a handful of annotated plans. Decide after step 1 gives real numbers.
5. Integration: the pipeline is a pure function `run_vector(pdf, cfg, classes, out_dir)`; call it as a subprocess from the
   Java/EJB side and read `rooms.json` / `elements.json` (schema documented in the README).
