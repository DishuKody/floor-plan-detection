# Floor Plan Detection and Segmentation — Implementation Brief for Claude

## Role and objective

Act as a senior computer-vision / machine-learning engineer. Analyze the supplied reference floor-plan image (and any accompanying PDF or source files) before choosing an implementation. Then build a reproducible, practical pipeline that detects and segments individual rooms and architectural elements in complex 2D floor plans.

The required output is not merely a room-type prediction or an attractive colored overlay. The system should produce:

1. A visualization with clearly distinguished room regions and architectural elements.
2. Individual room polygons in original-image coordinates.
3. Room-type labels and, where the model supports it, confidence or quality estimates.
4. Separate detections or masks for walls, doors, windows, and other supported elements.
5. Machine-readable JSON suitable for integration into an existing application.
6. Evaluation results that expose failure modes rather than hiding them behind one overall accuracy number.

**Do not assume a single pretrained model will solve every part of the problem.** Treat room geometry, room semantics, wall/door/window detection, and polygon construction as related but distinct tasks. Select a model combination only after inspecting the supplied drawing and testing feasible baselines.

## User constraints and environment

- Primary GPU: NVIDIA GeForce RTX 2050, **4 GB VRAM**.
- Integrated GPU: AMD Radeon 740M.
- Prefer local/offline processing and open-source components where practical.
- Avoid paid APIs or cloud GPU requirements as a default. If a cloud or paid option is genuinely worth considering, explain why and provide a local alternative.
- GPU memory is limited. Prefer inference first, mixed precision where supported, small batches, careful input sizing, and tiled/high-resolution strategies when technically appropriate.
- Do not assume the AMD integrated GPU can be used by CUDA/PyTorch. Treat NVIDIA CUDA as the likely GPU path on a supported NVIDIA driver/software stack; use CPU as a fallback. Explicitly check the actual operating system, driver, CUDA/PyTorch compatibility, and available memory before prescribing install commands.
- The user has already tried **CubiCasa5K** and found its accuracy insufficient. Do not simply repeat the same baseline or claim CubiCasa5K will solve the problem. It may be used only as a comparison, an annotation/data reference, or a source of transferable ideas after licensing and compatibility are checked.
- The supplied reference may be a dense architectural drawing with repeated room clusters, small bathrooms, corridors, communal spaces, doors, fixtures, and text/line annotations. Verify what is actually present in the provided image rather than treating this description as ground truth.
- If a PDF is supplied, determine whether it contains vector paths, raster images, or both. Do not assume the source is raster-only.

## Mandatory first step: inspect the actual reference image

Before selecting the model, open and inspect the supplied image at full useful resolution. If it is a PDF, inspect page count, page dimensions, embedded raster images, vector drawing paths, and text. Render pages only as needed for visual analysis, preserving aspect ratio and enough resolution to retain thin walls and small fixtures.

Write a short `analysis_report.md` (or an equivalent section in the final report) documenting:

- Input format, page/image dimensions, orientation, and whether the source contains vector geometry.
- Drawing characteristics: wall styles, line weights, room density, text/labels, door swings, windows, furniture, fixtures, repeated layouts, open-plan spaces, and image quality.
- Which room classes and architectural elements are visibly present.
- Likely failure modes for this particular image.
- Whether vector-first, raster-first, or hybrid processing is the best initial hypothesis, and why.
- Uncertainties that require visual review; do not invent labels or geometry that cannot be inferred.

Do not silently assume the image is a CAD export or that all walls are closed. Verify it. If the drawing has vector geometry, test whether it is useful for room reconstruction before rasterizing the entire page.

## Core functional requirements

### Room classes

Support a configurable class list, initially including:

- bedroom
- living_room
- kitchen
- bathroom
- toilet_or_wc (if distinguishable)
- dining_room
- corridor_or_hallway
- study_or_office
- storage
- laundry_or_utility
- balcony
- communal_or_shared_space
- other
- unknown

Not every model will support every class. Map model-native labels to this canonical list and use `unknown` when evidence is insufficient. Do not force a precise room type when the source is ambiguous. Allow the class list to be configured in a single file.

### Architectural elements

Where supported by the input and model, detect or segment:

- walls and wall boundaries
- doors and door swings
- windows
- stairs
- columns
- furniture or fixtures useful for room classification
- sanitary fixtures such as toilets, sinks, showers, and bathtubs
- kitchen counters / appliances
- text labels and dimensions, when relevant

Keep room regions and object detections in separate output layers. A door or fixture must not automatically be treated as a room boundary without geometric evidence.

### Required output artifacts

Produce at least:

- `outputs/overlay.png`: readable overlay on the original image, with a legend and different colors for room classes. Draw individual boundaries clearly. Avoid obscuring the original plan.
- `outputs/room_masks.png` or per-class masks: segmentation visualization.
- `outputs/rooms.json`: one record per room, with type, polygon, source coordinate frame, and available confidence/quality metadata.
- `outputs/elements.json`: walls/doors/windows/fixtures or other element records where implemented.
- `outputs/analysis_report.md`: input analysis, approach choice, dependencies, evaluation results, known limitations, and next steps.
- A runnable inference command and a README with setup, usage, troubleshooting, and hardware notes.
- Save intermediate model outputs and debug visualizations when useful; do not overwrite source files.

Use a normalized JSON schema similar to the following (extend it if needed, but document all fields):

```json
{
  "image": {
    "width": 2400,
    "height": 1800,
    "coordinate_origin": "top_left",
    "units": "pixels"
  },
  "rooms": [
    {
      "id": "room-0001",
      "type": "bathroom",
      "polygon": [[100.0, 100.0], [200.0, 100.0], [200.0, 180.0], [100.0, 180.0]],
      "confidence": null,
      "confidence_source": "model_probability_or_unavailable",
      "area_pixels": null,
      "review_required": true,
      "source": "model|vector|hybrid"
    }
  ],
  "elements": []
}
```

The numbers above are illustrative only, not detections from the user's plan. Use valid JSON in the actual output. Include any image resize, crop, padding, tile, rotation, or coordinate transforms so predictions can be mapped back accurately to original-image coordinates. If calibrated real-world coordinates are unavailable, explicitly state that coordinates are pixels and do not invent physical dimensions.

## Model selection: investigate, then decide

Compare the following practical directions against the actual input and the project's license / hardware constraints. Do not install every option. Select a small number for controlled evaluation.

### A. Polygon-first floor-plan reconstruction

Investigate **Raster2Seq** and its official implementation:
- Repository: https://github.com/Cornell-VAILab/Raster2Seq
- Paper/project information: https://cornell-vailab.github.io/Raster2Seq/

Assess:
- Checkpoints available and which datasets/classes they were trained on.
- Whether its output includes the room types and geometry required here.
- Exact inference and preprocessing requirements.
- Python/PyTorch/CUDA versions, memory use, and whether any components need compilation.
- License for code, checkpoints, and training data.
- Whether inference can run at a useful resolution within 4 GB VRAM.
- Whether its polygon output preserves small bathrooms and narrow corridors in this reference.

Do not rely on reported benchmark scores as a prediction of accuracy on this image. Evaluate it directly.

### B. Vector-first extraction (especially for PDF/CAD-origin plans)

If the supplied PDF contains useful vector paths, investigate extracting them before raster-only processing.

Potential tools:
- PyMuPDF: https://github.com/pymupdf/PyMuPDF
- Shapely: https://github.com/shapely/shapely
- OpenCV: https://github.com/opencv/opencv
- ezdxf for a separately supplied DXF file: https://github.com/mozman/ezdxf

Prototype:
1. Extract PDF path/line/curve geometry and metadata.
2. Render the page at high resolution for visual reference.
3. Identify likely wall paths using geometry, line weight, continuity, and spatial context.
4. Remove or down-weight dimensions, grids, furniture, hatches, and annotations.
5. Join wall segments and bridge only small justified gaps.
6. Build candidate enclosed regions; preserve openings and avoid merging rooms across doors or open-plan connections.
7. Validate candidate polygons and visualize every intermediate step.
8. Classify candidate regions using a separate model, OCR, fixture evidence, and layout context.

Do not treat every closed contour as a room. Do not indiscriminately close gaps or fill all regions. Vector-first processing is a hypothesis to test, not a guaranteed solution.

### C. Semantic / instance segmentation

Investigate a modern segmentation architecture such as:
- SegFormer: https://github.com/NVlabs/SegFormer
- Mask2Former: https://github.com/facebookresearch/Mask2Former
- DeepFloorplan as a floor-plan-specific research baseline: https://github.com/zlzeng/DeepFloorplan

For each candidate, verify current maintenance status, license, checkpoint availability, dependencies, and supported classes. A general segmentation checkpoint is not automatically a floor-plan model.

If using semantic segmentation, add instance separation for distinct rooms of the same class. Candidate post-processing includes connected components, wall-guided separation, watershed or distance-transform methods where appropriate, contour extraction, and Shapely geometry validation. Avoid relying on connected components alone when adjacent same-class rooms touch in the mask.

### D. Foundation-model / prompt-assisted segmentation

Investigate SAM-family models only if useful after inspecting the image. Generic segmentation masks do not inherently provide room semantics or reliable room boundaries. Any SAM-based approach needs a strategy for prompts, candidate region generation, room classification, and boundary validation.

A few-shot or prompt-assisted approach may be useful when labels are scarce, but do not assume that a handful of examples provides production-level room-type accuracy. Check model licenses, local memory requirements, and whether text/image APIs would send confidential floor plans externally. Default to local inference.

### E. Object detection and OCR as supporting signals

Use these as supporting evidence, not as the only room detector:
- OCR for explicit labels such as BEDROOM, KITCHEN, WC, or BATH.
- Object detection/segmentation for toilets, sinks, showers, bathtubs, kitchen fixtures, doors, and windows.
- Geometry and spatial association to link labels/fixtures to candidate rooms.

A generic vision-language model may help propose labels but should not be trusted to generate final precise polygons without geometric validation.

### F. Hybrid pipeline

Prefer testing a hybrid approach when the image warrants it:

1. Recover vector geometry if available.
2. Render high-resolution raster tiles with controlled overlap.
3. Run a polygon-first or segmentation model.
4. Extract/repair candidate regions with geometric rules.
5. Detect doors, windows, fixtures, and text as separate layers.
6. Reconcile geometry with semantic masks and labels.
7. Map all outputs back to the original coordinate frame.
8. Calculate quality indicators and flag uncertain rooms for manual review.

Use only components that demonstrate measurable benefit on the reference and validation set. Avoid unnecessary complexity.

## Hardware-aware implementation

### Before installing dependencies

Inspect the existing environment:
- OS and version
- Python version
- NVIDIA driver version
- Whether `nvidia-smi` works
- Installed PyTorch version and `torch.cuda.is_available()`
- GPU memory available at inference time
- Available RAM and disk space

Choose a compatible PyTorch/CUDA wheel using the official installation selector/documentation:
- https://pytorch.org/get-started/locally/

Do not assume the system's reported CUDA driver capability is the same as the CUDA runtime bundled with a PyTorch wheel. Do not install an arbitrary CUDA toolkit unless it is actually needed.

### GPU constraints

- Start with pretrained inference, not training from scratch.
- Use batch size 1 initially.
- Use `torch.inference_mode()` for inference.
- Test FP16/autocast only if the selected model supports it correctly; compare output quality with FP32.
- Release unused tensors and avoid holding multiple full-resolution copies of the image on the GPU.
- Use CPU preprocessing and post-processing where practical.
- Use tiled inference with overlap if required, while handling duplicate detections and stitching boundaries carefully.
- Avoid excessive downscaling. Small bathrooms, thin walls, and door symbols are high-priority details.
- If fine-tuning, begin with a small batch, gradient accumulation if useful, frozen backbone or partial fine-tuning if appropriate, and checkpointing if supported. Measure memory instead of assuming these will fit.
- Treat AMD Radeon 740M as a possible CPU/iGPU acceleration option only after checking the actual OS and framework support. Do not present it as a CUDA device.
- Provide CPU fallback and an early, clear error if a required checkpoint or GPU dependency is missing.

## Practical implementation phases

### Phase 0 — Inspect and report
- Inspect the actual reference image/PDF first.
- Determine whether vector geometry is available.
- Produce `analysis_report.md`.
- Identify the 3–5 most important failure modes to measure.

### Phase 1 — Establish a reproducible baseline
- Build image/PDF loading and high-resolution rendering.
- Run one selected pretrained polygon model if compatible.
- Save raw output, overlay, JSON, timing, peak VRAM use, and errors.
- Do not fine-tune yet.

### Phase 2 — Test vector extraction
- If vector paths exist, prototype candidate room extraction.
- Visualize extracted lines and regions separately.
- Compare vector-derived regions with model-derived polygons.
- Retain this branch only if it improves measured boundary quality or reduces errors.

### Phase 3 — Compare a segmentation baseline
- Select one feasible segmentation architecture or floor-plan-specific model.
- Use the same test images and coordinate conventions.
- Convert predictions into individual room instances and polygons.
- Record per-class and boundary metrics.

### Phase 4 — Hybrid reconciliation
- Combine only complementary signals: vector boundaries, predicted masks/polygons, labels, and fixtures.
- Add rules for invalid polygons, tiny fragments, overlaps, and merged rooms.
- Keep rules conservative and configurable; do not silently delete plausible small rooms.

### Phase 5 — Fine-tuning, if justified
- If all pretrained approaches fail, prepare annotations on representative plans.
- Split train/validation/test by building or project, not random crops of the same plan.
- Fine-tune a suitable pretrained model; do not train from scratch unless there is a compelling reason.
- Track whether each training iteration improves held-out performance.

### Phase 6 — Final validation and handoff
- Run the full pipeline from a clean environment.
- Verify outputs and coordinate transforms.
- Document supported classes, known failure cases, hardware usage, and exact commands.
- Provide a concise recommendation explaining why the chosen solution beat the alternatives tested.

## Code organization

Use a modular structure similar to this, adapting it to the chosen approach rather than creating empty modules:

```text
floor-plan-detection/
├── README.md
├── analysis_report.md
├── requirements.txt             # or environment.yml/pyproject.toml
├── config/
│   ├── classes.yaml
│   └── inference.yaml
├── src/
│   ├── cli.py                   # command-line entry point
│   ├── input/
│   │   ├── load_image.py
│   │   └── pdf_inspection.py
│   ├── preprocessing/
│   │   ├── render_pdf.py
│   │   ├── normalize_image.py
│   │   └── tiling.py
│   ├── models/
│   │   ├── base.py
│   │   └── selected_model.py
│   ├── vector/
│   │   ├── extract_pdf_paths.py
│   │   └── candidate_regions.py
│   ├── postprocessing/
│   │   ├── instances.py
│   │   ├── polygons.py
│   │   └── geometry_validation.py
│   ├── supporting_signals/
│   │   ├── ocr.py
│   │   └── fixtures.py
│   ├── output/
│   │   ├── json_writer.py
│   │   └── visualization.py
│   └── evaluation/
│       ├── metrics.py
│       └── report.py
├── tests/
│   ├── test_coordinates.py
│   ├── test_polygon_validity.py
│   └── test_json_schema.py
└── outputs/
    ├── overlay.png
    ├── rooms.json
    └── elements.json
```

Do not create dummy implementations that claim to detect rooms. Every module included in the final deliverable must either work or be clearly marked as a not-yet-implemented optional component.

## Dependencies

Choose dependencies based on the selected model, and pin versions that are compatible with the environment. Likely general-purpose dependencies include:

- `numpy`
- `Pillow`
- `opencv-python`
- `PyMuPDF`
- `shapely`
- `scikit-image`
- `matplotlib`
- `PyYAML`
- `pytest`
- `torch` and `torchvision` when required by the chosen model
- An OCR library only if label extraction demonstrates value
- Model-specific dependencies only when necessary

Do not blindly install all possible frameworks. Raster2Seq, DeepFloorplan, and segmentation frameworks may require incompatible versions of Python, PyTorch, CUDA, or other libraries. Isolate incompatible experiments in separate environments if needed. Document the exact tested versions and commands.

## Evaluation strategy

### Ground truth
Create or obtain manually reviewed room polygons and class labels for a small but representative validation set. Include difficult examples from the reference drawing. If annotation effort is limited, begin with a carefully selected set of representative rooms and expand it as failure modes become clear.

### Required metrics
Report:
- Per-class precision, recall, and F1 for room instances.
- Per-class IoU or Dice for segmentation masks.
- Polygon/room matching protocol, including any IoU threshold used.
- Boundary accuracy or boundary F-score where practical.
- Counts of missed rooms, merged rooms, split rooms, false rooms, and incorrect labels.
- Invalid polygon count, self-intersections, and major overlaps.
- Runtime per page and peak GPU VRAM use.
- Results separately for small rooms, dense clusters, corridors, and communal/open spaces where possible.

Do not compare results from different datasets or evaluation protocols as if they were directly comparable. Report model-reported scores separately from your own measured results.

### Visual review
Generate side-by-side panels:
1. Original image.
2. Raw model output.
3. Predicted polygons/instances.
4. Final colored overlay.
5. Error visualization against annotations, if ground truth is available.

Use distinct colors and a legend, but retain the original linework. Make room boundaries legible at useful zoom levels. Save per-tile visualizations when tiling is used.

### Acceptance criteria
Before evaluation, define application-specific thresholds. At minimum, the pipeline must not be considered successful solely because it creates a colorful image. It must identify individual spaces, preserve small rooms, provide stable coordinate mapping, and expose uncertain predictions for review.

## Engineering rules

- Do not claim accuracy improvements without measurements.
- Do not fabricate confidence scores. If confidence is unavailable or uncalibrated, use `null` or clearly label it as a heuristic.
- Do not invent real-world dimensions from pixel coordinates.
- Do not silently alter the original input.
- Preserve a reversible mapping from model/tile coordinates to original-image coordinates.
- Handle image rotation, scaling, padding, and crop offsets explicitly.
- Validate all JSON and polygon geometry.
- Include useful logs, deterministic seeds where practical, and clear error messages.
- Keep raw outputs so that post-processing improvements can be compared.
- Check licenses for code, pretrained weights, and datasets separately, especially before commercial use.
- Never upload private floor plans to external APIs without explicit approval.
- Prefer the simplest approach that demonstrates measurable accuracy on representative data.
- If a candidate model cannot be installed or run in the available environment, document the blocker and test the next feasible candidate instead of pretending it worked.

## Expected final response from Claude

At completion, provide:

1. **Image analysis:** What the actual plan contains and what makes it difficult.
2. **Decision rationale:** Why the chosen approach is better suited than the alternatives tested.
3. **Implemented files:** List the files created and their purpose.
4. **Exact setup commands:** Commands tested on the available OS and hardware.
5. **Exact inference command:** One command that runs the pipeline against the supplied reference file.
6. **Outputs:** Paths to the overlay, masks, room polygons, and element detections that were actually generated.
7. **Evaluation:** Measured metrics if ground truth is available; otherwise clearly state that quantitative accuracy is not yet established.
8. **Hardware usage:** Observed GPU/CPU, runtime, and peak VRAM where measurable.
9. **Known limitations:** Specific rooms or elements the system still misses or confuses.
10. **Next steps:** The smallest next experiment likely to improve accuracy.

Start by inspecting the supplied reference image and its source file. Do not begin by blindly installing a model or generating a generic script. Make evidence-based choices, implement the smallest meaningful experiment, run it, inspect the visual output, and iterate.
