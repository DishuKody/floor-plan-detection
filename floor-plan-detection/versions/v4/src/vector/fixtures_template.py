"""Repeated-symbol (template) detection on the thin-line layer: finds copies of an exemplar symbol,
including mirrored / rotated ones. CAD blocks are identical copies, so matches are near-exact (score ~1.0)."""
import numpy as np
import cv2


def line_layer(paths, roi, scale, w_range=(0.1, 0.3), skip=()):
    x0, y0, x1, y1 = roi
    L = np.zeros((int((y1 - y0) * scale), int((x1 - x0) * scale)), np.uint8)
    for i, q in enumerate(paths):
        if i in skip or q.width == 0 or not (w_range[0] <= q.width <= w_range[1]):
            continue
        cv2.polylines(L, [((q.pts - [x0, y0]) * scale).astype(np.int32)], q.closed, 255, 1)
    return L


def _variants(T):
    out = []
    for k in range(4):
        R = np.rot90(T, k)
        for flip in (None, 1):
            out.append(np.ascontiguousarray(R if flip is None else cv2.flip(R, flip)))
    seen, uniq = [], []
    for v in out:   # drop duplicate variants (symmetric symbols)
        if not any(v.shape == s.shape and np.array_equal(v, s) for s in seen):
            seen.append(v); uniq.append(v)
    return uniq


def find_symbol(L, roi, scale, exemplar_box_pt, search_boxes_pt, thresh=0.85, min_dist_pt=20.0, blur=3, min_ink=0.7, device="cpu"):
    """Return matches as dicts with box_pt=[x0,y0,x1,y1] in page points."""
    ex = exemplar_box_pt
    T = L[int((ex[1] - roi[1]) * scale):int((ex[3] - roi[1]) * scale), int((ex[0] - roi[0]) * scale):int((ex[2] - roi[0]) * scale)]
    ink_T = float((T > 0).sum())
    soft = lambda a: cv2.GaussianBlur(cv2.dilate(a, np.ones((2, 2), np.uint8)).astype(np.float32), (0, 0), blur)
    cands = []
    for sb in search_boxes_pt:
        ox, oy = int((sb[0] - roi[0]) * scale), int((sb[1] - roi[1]) * scale)
        crop = L[oy:int((sb[3] - roi[1]) * scale), ox:int((sb[2] - roi[0]) * scale)]
        Ls = soft(crop)
        corr = None
        if device == "cuda":
            from ..gpu import FftCorrelator
            corr = FftCorrelator(Ls, "cuda")       # image FFT + integral images computed once for all template variants
        ink = cv2.integral((crop > 0).astype(np.uint8))
        for Tv in _variants(T):
            h, w = Tv.shape
            if h >= crop.shape[0] or w >= crop.shape[1]:
                continue
            if corr is not None:
                R = corr.match(soft(Tv))
            else:
                R = cv2.matchTemplate(Ls, soft(Tv), cv2.TM_CCOEFF_NORMED)
            ys, xs = np.nonzero(R > thresh)
            for y, x in zip(ys, xs):
                n = ink[y + h, x + w] - ink[y, x + w] - ink[y + h, x] + ink[y, x]
                if n < min_ink * ink_T:      # blank windows give meaningless correlations
                    continue
                cands.append((float(R[y, x]), (ox + x, oy + y, ox + x + w, oy + y + h)))
    cands.sort(reverse=True)
    kept = []
    for sc, b in cands:
        c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) / scale + np.array(roi[:2])
        if all(np.linalg.norm(c - np.array(m["center_pt"])) > min_dist_pt for m in kept):
            kept.append(dict(score=round(sc, 3), center_pt=c.tolist(),
                             box_pt=[b[0] / scale + roi[0], b[1] / scale + roi[1], b[2] / scale + roi[0], b[3] / scale + roi[1]]))
    return kept
