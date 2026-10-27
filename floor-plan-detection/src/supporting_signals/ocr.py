"""Tiled local OCR (RapidOCR / ONNX Runtime) over a rendered PDF page region. Returns text boxes in page points."""
import unicodedata
import numpy as np
import pymupdf
import cv2


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def _providers(eng):
    """Execution providers of every ONNX session inside the RapidOCR engine as actually created (ORT silently falls back to CPU)."""
    import onnxruntime as ort
    out, seen = {}, set()

    def walk(o, path, depth=0):
        for k, v in vars(o).items():
            if isinstance(v, ort.InferenceSession):
                out[f"{path}.{k}".lstrip(".")] = list(v.get_providers())
            elif hasattr(v, "__dict__") and id(v) not in seen and depth < 3:
                seen.add(id(v))
                walk(v, f"{path}.{k}", depth + 1)
    walk(eng, "")
    return out


def _patch_cudnn_search(mode):
    """RapidOCR hardcodes cudnn_conv_algo_search=EXHAUSTIVE; with dynamic input shapes (every text line has its own width) cuDNN
    re-benchmarks algorithms for each new shape. HEURISTIC avoids that."""
    from rapidocr_onnxruntime.utils import infer_engine as ie
    if getattr(ie.OrtInferSession, "_patched_cudnn", None) == mode:
        return
    orig = getattr(ie.OrtInferSession, "_orig_get_ep_list", ie.OrtInferSession._get_ep_list)
    ie.OrtInferSession._orig_get_ep_list = orig

    def patched(self):
        return [(n, {**o, "cudnn_conv_algo_search": mode} if n == "CUDAExecutionProvider" else o) for n, o in orig(self)]
    ie.OrtInferSession._get_ep_list = patched
    ie.OrtInferSession._patched_cudnn = mode


def run_ocr(page: pymupdf.Page, roi, cfg, device="cpu", info=None):
    if device == "cuda":
        import torch  # noqa: F401  - loads the CUDA 12.x / cuDNN 9 DLLs that onnxruntime-gpu reuses
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    from rapidocr_onnxruntime import RapidOCR
    use = device == "cuda"
    if use and cfg.get("cudnn_algo_search"):
        _patch_cudnn_search(cfg["cudnn_algo_search"])
    eng = RapidOCR(det_use_cuda=use, cls_use_cuda=use, rec_use_cuda=use)
    if info is not None:
        info["ocr_providers"] = _providers(eng)
    up, T, ov = cfg["upscale"], cfg["tile_px"], cfg["overlap_px"]
    x0, y0, x1, y1 = roi
    pix = page.get_pixmap(matrix=pymupdf.Matrix(up, up), clip=pymupdf.Rect(x0, y0, x1, y1), colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    H, W = img.shape
    out = []
    step = T - ov
    for ty in range(0, max(1, H - ov), step):
        for tx in range(0, max(1, W - ov), step):
            tile = img[ty:ty + T, tx:tx + T]
            res, _ = eng(cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR))
            for box, text, score in (res or []):
                score = float(score)
                if score < cfg["min_score"] or not text.strip():
                    continue
                b = np.array(box, float)
                bb = [(b[:, 0].min() + tx) / up + x0, (b[:, 1].min() + ty) / up + y0, (b[:, 0].max() + tx) / up + x0, (b[:, 1].max() + ty) / up + y0]
                out.append(dict(text=unicodedata.normalize("NFKC", text).strip(), score=score, bbox=bb))
    # de-duplicate detections from overlapping tiles
    out.sort(key=lambda r: -r["score"])
    kept = []
    for r in out:
        if not any(_iou(r["bbox"], k["bbox"]) > 0.5 for k in kept):
            kept.append(r)
    return kept
