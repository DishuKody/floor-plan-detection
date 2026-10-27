"""Candidate room regions from a wall mask: close door openings, fill wall channels, label free space."""
import numpy as np
import cv2


def close_doors(mask, arcs, roi, scale, probe=(2.0, 9.0)):
    """Draw a closing chord across each door opening. The opening is the hinge->arc-end radius that lies along the wall line
    (i.e. the wall mask continues on the far side of the hinge)."""
    x0, y0 = roi[:2]
    chords = []
    for a in arcs:
        best = None
        for end in (a.p_start, a.p_end):
            u = (end - a.center) / (np.linalg.norm(end - a.center) + 1e-9)
            hits = 0
            for d in np.linspace(probe[0], probe[1], 8):
                p = ((a.center - u * d) - [x0, y0]) * scale
                xi, yi = int(p[0]), int(p[1])
                if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]:
                    hits += int(mask[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2].any())
            if best is None or hits > best[0]:
                best = (hits, end)
        chords.append((a.center, best[1], best[0]))
    for c, e, h in chords:
        p0 = tuple(int(v) for v in (c - [x0, y0]) * scale)
        p1 = tuple(int(v) for v in (e - [x0, y0]) * scale)
        cv2.line(mask, p0, p1, 255, 3)
    return chords


def bridge_gaps(mask, scale, gap_pt):
    """Axis-aligned directional closing: bridges collinear wall gaps up to gap_pt (dangling partitions, hairline openings)."""
    L = int(gap_pt * scale) | 1
    h = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    v = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    return cv2.bitwise_or(mask, cv2.bitwise_or(h, v))


def label_regions(mask, scale, wall_close_pt=3.5, min_area_pt2=30.0, min_inscribed_pt=0.0, gap_pt=0.0):
    k = int(wall_close_pt * scale) | 1
    if gap_pt > 0:
        mask = bridge_gaps(mask, scale, gap_pt)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    free = (closed == 0).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(free, connectivity=4)
    H, W = free.shape
    regions = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        touches = x == 0 or y == 0 or x + w >= W or y + h >= H
        area_pt2 = a / scale ** 2
        if area_pt2 < min_area_pt2:
            continue
        sub = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        insc = float(cv2.distanceTransform(np.pad(sub, 1), cv2.DIST_L2, 3).max()) / scale
        if insc < min_inscribed_pt:
            continue
        regions.append(dict(label=i, area_pt2=area_pt2, bbox=(x, y, w, h), border=bool(touches), inscribed_pt=insc))
    return closed, lab, regions
