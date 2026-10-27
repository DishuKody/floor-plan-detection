"""Second pass for named structural spaces (stair, lift, riser, AOV): re-segment using only thick structural walls + door chords,
so symbol linework inside the enclosure (treads, 'X' shaft marks, hatch) cannot fragment the room."""
import numpy as np
import cv2
from shapely.geometry import Point
from .polygons import region_polygon


def structural_rooms(thick, chords, texts, keywords, roi, scale, pt2_per_m2, existing, area_m2=(1.0, 60.0), close_pt=3.5):
    """keywords: list of lowercase, space-free substrings that mark a structural space. chords: [(c, e, ...)] page pt.
    existing: list of shapely polygons of current rooms. Returns list of (polygon, replaced_indices, label_text)."""
    t = thick.copy()
    x0, y0 = roi[:2]
    for c, e, *_ in chords:
        cv2.line(t, tuple(int(v) for v in (c - [x0, y0]) * scale), tuple(int(v) for v in (e - [x0, y0]) * scale), 255, 3)
    k = int(close_pt * scale) | 1
    t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((t == 0).astype(np.uint8), connectivity=4)
    H, W = t.shape
    out, seen = [], set()
    for tx in texts:
        txt = "".join(tx["text"].lower().split())
        if not any(kw in txt for kw in keywords) or "type" in txt:
            continue
        cx, cy = (tx["bbox"][0] + tx["bbox"][2]) / 2, (tx["bbox"][1] + tx["bbox"][3]) / 2
        px, py = int((cx - x0) * scale), int((cy - y0) * scale)
        if not (0 <= px < W and 0 <= py < H):
            continue
        lid = lab[py, px]
        if lid == 0:        # label sits on a wall pixel: take the nearest free pixel within 5 pt
            r = int(5 * scale)
            win = lab[max(py - r, 0):py + r + 1, max(px - r, 0):px + r + 1]
            ys, xs = np.nonzero(win)
            if len(xs) == 0:
                continue
            j = np.argmin((xs - min(px, r)) ** 2 + (ys - min(py, r)) ** 2)
            lid = win[ys[j], xs[j]]
        if lid in seen:
            continue
        x, y, w, h, a = stats[lid]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        am2 = a / scale ** 2 / pt2_per_m2
        if not (area_m2[0] <= am2 <= area_m2[1]):
            continue
        poly = region_polygon(lab, lid, (x, y, w, h), roi, scale)
        if poly is None:
            continue
        # must not swallow a room that is larger than itself
        if any(e.area > poly.area and e.intersection(poly).area > 0.1 * poly.area for e in existing):
            continue
        seen.add(lid)
        replaced = [i for i, e in enumerate(existing) if e.intersection(poly).area > 0.5 * e.area]
        out.append((poly, replaced, tx["text"]))
    return out
