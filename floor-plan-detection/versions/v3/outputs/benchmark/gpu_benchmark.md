| configuration | pipeline s median (min-max) | process wall s median | OCR stage s | pod+symbol matching s | peak RAM MB | OCR providers |
|---|---|---|---|---|---|---|
| CPU-only env (.venv_cpuref: no PyTorch, onnxruntime 1.31) | 42.4 (41.1-54.5) | 43.91 | 15.49 | 4.69 | 1004 | CPUExecutionProvider |
| GPU env, --device cpu | 43.4 (41.9-50.0) | 44.79 | 15.98 | 4.79 | 996 | CPUExecutionProvider |
| GPU env, --device cuda (RTX 2050) | 30.6 (29.8-37.3) | 34.6 | 7.3 | 1.86 | 2070 | CPUExecutionProvider, CUDAExecutionProvider |

Speed-up CUDA vs CPU-only env (medians of 5 interleaved rounds): pipeline x1.39, process wall x1.27, OCR stage x2.12.
GPU measured (separate nvidia-smi-sampled run, 44 samples @ 0.25 s): baseline VRAM used 405 MB, peak 1674 MB (+1269 MB of 4096), peak utilisation 84 %, mean 12.4 %.
CPU vs CUDA result identity: {"n_rooms": [109, 109], "room_types_identical": true, "room_polygons_identical": true, "windows": [62, 62], "doors": [105, 105], "ocr_boxes": [132, 132], "ocr_text_sets_identical": true, "ocr_texts_only_a": [], "ocr_texts_only_b": []}
Conditions: laptop on battery (Balanced plan, CPU ~2.5 GHz): absolute times drift, so configurations were interleaved round-robin. Round-1 CUDA process wall (92.7 s) is the first-ever cold start (DLL / disk cache).
