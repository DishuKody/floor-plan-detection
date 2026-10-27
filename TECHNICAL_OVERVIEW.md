# Floor-Plan Detection System: Technical Overview

**Audience:** technical stakeholders and management  
**Status:** working system, version 5 (verified by an automated test suite and by evaluation runs on four reference drawings)  
**Classification:** Confidential. This document describes *what* the system uses and delivers, not *how* its proprietary processing works.

---

## 1. Purpose

The system reads architectural floor-plan drawings supplied as PDF files and produces **structured geometric data**: the outline of each room and space, classified by type, together with architectural elements (doors, windows, columns, walls, circular shapes) and visual overlays for review. The output is designed to be consumed by other systems; the intended integration target is a Java/EJB backend that invokes the tool and reads its JSON output.

## 2. Capabilities at a Glance

| Capability | Description |
|---|---|
| **Input formats** | PDF floor plans, both CAD-exported (vector) drawings and scanned/embedded-image (raster) drawings. The system selects the appropriate processing mode automatically. |
| **Drawing range** | Small single-storey plans through to very large, A0-size, multi-wing building plans. |
| **Room / space detection** | Closed polygons for rooms and spaces, including irregular and curved shapes, with shape descriptors (circle, square, rectangle, rectilinear polygon, irregular polygon). |
| **Room classification** | Rule-based typing from text labels and geometric evidence (for example bedroom, bathroom, kitchen, corridor, classroom, hall, stair, lift), with the evidence recorded for each decision and uncertain results flagged for review. |
| **Element detection** | Doors, windows, columns, walls and circular shapes, exported as structured elements. |
| **Text reading** | Local optical character recognition (OCR) to read room names and unit identifiers printed on the drawing. |
| **Scale awareness** | Dimensions and areas are reported using the drawing's printed scale. |
| **Full-page, undistorted output** | Overlays and masks always cover the entire page at its true aspect ratio; nothing is cropped or stretched. Coordinates are provided in several consistent reference frames (page pixels, PDF points, source-image pixels where applicable). |
| **Hardware acceleration** | Optional NVIDIA GPU acceleration for selected stages; the system runs identically on CPU-only machines. |
| **Offline operation** | All processing, including OCR, runs locally. No cloud service or external API is called at run time. |

## 3. Technology Stack

### 3.1 Language and runtime
- **Python 3.11** for the entire processing system.
- **Windows batch scripts** (`setup.bat`, `run.bat`) for installation and command-line execution on Windows.
- Packaged as a **command-line tool**, which makes it straightforward to call from other applications (including Java services).

### 3.2 Core libraries
| Area | Technology |
|---|---|
| PDF parsing and page rendering | **PyMuPDF** |
| Image processing and computer vision | **OpenCV** (headless build) |
| Numerical computing | **NumPy**, **SciPy** |
| Image analysis | **scikit-image** |
| Geometry (polygons, validity, set operations) | **Shapely** (GEOS) |
| Polygon clipping utilities | **pyclipper** |
| Text recognition | **RapidOCR** on **ONNX Runtime** (local models, bundled with the package) |
| Configuration | **YAML** (PyYAML) |
| Imaging utilities | **Pillow** |
| System monitoring | **psutil**, NVIDIA `nvidia-smi` sampling for GPU measurements |
| Testing | **pytest** |

### 3.3 Hardware acceleration
- **NVIDIA CUDA** through **PyTorch** (CUDA 12.8 build) and **ONNX Runtime GPU** (CUDA execution provider).
- GPU acceleration is **optional**. The system detects the available device at start-up (`auto`, `cpu` or `cuda`), falls back to CPU automatically, and records in its run statistics which execution providers were actually used.
- Tested on an NVIDIA RTX 2050 (4 GB) laptop GPU. Only part of the pipeline benefits from the GPU (OCR and symbol matching); the remaining stages are CPU-bound, so overall gains are moderate (see section 6).

### 3.4 Machine learning
- The only neural-network inference in the system is the **OCR model set** (text detection, orientation classification, text recognition) executed through ONNX Runtime.
- Room and element detection is **classical computational geometry and image processing**; it does not depend on a trained segmentation model. No claim is made of learned room detection.

### 3.5 Packaging and portability
- Reproducible environment via pinned dependency files (CPU and GPU variants) and a one-step setup script.
- Runs on machines without a GPU. Tested on Windows 11; Linux and macOS have not been tested.

## 4. System Architecture (High Level)

The system is organised as a small number of cooperating components:

```
            ┌───────────────┐
 PDF  ───►  │ Input handling │  detects drawing type (vector / raster)
            └───────┬───────┘
          ┌─────────┴──────────┐
          ▼                    ▼
  ┌───────────────┐    ┌───────────────┐
  │ Vector engine │    │ Raster engine │
  └───────┬───────┘    └───────┬───────┘
          └─────────┬──────────┘
                    ▼
        ┌────────────────────────┐      ┌──────────────┐
        │ Shape & room analysis   │◄────►│ Text reading │
        │ and classification      │      │   (OCR)      │
        └───────────┬────────────┘      └──────────────┘
                    ▼
        ┌────────────────────────┐
        │ Output generation       │  JSON · overlays · masks · run statistics
        └────────────────────────┘
```

| Component | Role |
|---|---|
| **Input handling** | Opens the PDF and decides whether it contains usable vector drawing data or only an embedded image. |
| **Vector engine** | Works directly from the drawing's vector content for CAD-exported plans, adapting to each sheet's conventions and printed scale. |
| **Raster engine** | Works from image content for scanned or picture-based plans. |
| **Text reading (OCR)** | Reads labels and identifiers from the drawing; runs locally and can use the GPU. |
| **Shape & room analysis** | Produces validated polygons and shape descriptors, and assigns a room type with supporting evidence and review flags. |
| **Configuration and sheet profiles** | YAML-based configuration. Sheet-specific information that cannot be derived from a drawing (for example its printed scale) is supplied through optional profiles. |
| **Output generation** | Writes structured data and visual deliverables. |
| **Evaluation tooling** | Regression, benchmark and quality-proxy utilities used to verify the system (not required at run time). |

## 5. Outputs

For each processed PDF the system produces:

| Output | Content |
|---|---|
| `<name>_rooms.json` | Room polygons (with holes where present), type, shape descriptors, approximate area, text found inside, supporting evidence, review flags. Coordinates in page pixels and PDF points (and source-image pixels for raster inputs). |
| `<name>_elements.json` | Doors, windows, columns, walls, circles and other detected elements with their geometry. |
| `<name>_overlay.png` | Full-page rendering with detected rooms and elements drawn on top, including a legend. |
| `<name>_masks.png` | Full-page colour-coded room mask. |
| `run_stats.json` and `debug/` | Device and execution-provider information, timing, memory use, OCR output and diagnostic images. |

The JSON files carry the full coordinate transform and calibration information so that downstream systems can map results back to the original drawing.

## 6. Verified Performance and Quality

All figures below were measured on the project's four reference drawings (a student-accommodation floor plan, two simple raster plans, and a large A0 school plan). They are reported as evidence of capability, not as guarantees for arbitrary drawings.

**Coverage and stability**
- The complex CAD plan is reproduced identically across versions (all 109 rooms, same types, geometry unchanged).
- On the simple raster plans, detected boundaries were compared with a hand-annotated reference (13 marked boundary lines): mean distance 5.3 px at the evaluated resolution, 84 % of the marked boundary pixels within 8 px.
- On the very large school plan, the system finds the whole building (259 rooms/spaces) and locates 35 of 38 printed room names inside a detected room (the previous version: 8 of 38).

**Runtime (laptop, NVIDIA RTX 2050)**

| Drawing | Time |
|---|---|
| Simple raster plans | about 4 to 6 s |
| Complex CAD plan (A1) | about 16 s |
| Very large CAD plan (A0) | about 1 minute |

On the A0 plan, GPU acceleration made the whole pipeline about 1.3 times faster and the OCR stage about 2.5 times faster than a CPU-only setup, with identical room results. Peak GPU memory use was about 1.7 GB.

**Quality assurance**
- Automated test suite of 68 tests, including synthetic plans with known geometry, coordinate-consistency checks and regression checks.
- Evaluation utilities for version-to-version regression, reference-line comparison and CPU/GPU benchmarking.

## 7. Known Limitations

Stated plainly so that expectations are accurate:

- **No full ground truth.** Accuracy has been assessed against a limited hand-drawn reference and printed room labels; there is no complete annotated dataset, so formal precision/recall figures for all room types are not available.
- **Open-plan spaces.** Where a drawing contains no separator between spaces, boundaries are approximated and flagged for human review.
- **Room typing is rule-based**, relies on legible labels, and is reported with evidence rather than as a calibrated probability. Some spaces remain unclassified.
- **Printed scale** must be supplied for vector sheets that are not at 1:100, because it is not machine-readable in the drawings tested.
- **Sheet-specific symbol libraries** (for example particular CAD symbol blocks) need a sheet profile to be recognised.
- **OCR dependence.** Text that the OCR cannot read cannot contribute to classification.
- **GPU benefit is limited** to the stages that use it; very large drawings remain dominated by CPU processing.
- **Platform.** Tested on Windows 11 only.

## 8. Integration

- **Interface:** command-line invocation with an input PDF, an output folder, an optional device selection and an optional sheet profile; results are plain JSON and PNG files.
- **Java/EJB backend:** the intended pattern is to launch the tool as an external process and read the JSON outputs. A direct service/API wrapper has not been built.
- **Operations:** a one-step setup script installs the environment; the tool needs network access only at installation time.

## 9. Intellectual Property Notice

The processing methods, decision rules, thresholds, parameter values, processing sequence and optimisation techniques of this system are proprietary and confidential. This document intentionally describes only the technology stack and observable capabilities. It must not be redistributed outside the organisation without approval, and it should not be interpreted as a specification from which the system could be reproduced.
