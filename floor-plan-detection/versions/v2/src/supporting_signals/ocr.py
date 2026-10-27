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


def run_ocr(page: pymupdf.Page, roi, cfg):
    from rapidocr_onnxruntime import RapidOCR
    eng = RapidOCR()
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
