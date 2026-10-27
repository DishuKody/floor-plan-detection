"""Instance separation: split a free-space region that holds several labelled units (merged rooms) with a
marker-controlled watershed on the distance transform. Cuts fall on necks/saddles between the label seeds."""
import numpy as np
import cv2
from skimage.segmentation import watershed
from .polygons import region_polygon


def split_region(lab, label, bbox, seeds_pt, roi, scale, simplify_pt=0.4):
    """Return one polygon per seed (page points), or None if a seed falls outside the region."""
    x, y, w, h = bbox
    reg = (lab[y:y + h, x:x + w] == label)
    dt = cv2.distanceTransform(np.pad(reg.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    markers = np.zeros(reg.shape, np.int32)
    for i, (sx, sy) in enumerate(seeds_pt, 1):
        px, py = int((sx - roi[0]) * scale) - x, int((sy - roi[1]) * scale) - y
        if not (0 <= px < w and 0 <= py < h):
            return None
        # a label may sit on a wall pixel: snap the seed to the nearest free pixel within 3pt
        if not reg[py, px]:
            ys, xs = np.nonzero(reg[max(py - 12, 0):py + 13, max(px - 12, 0):px + 13])
            if len(xs) == 0:
                return None
            k = np.argmin((xs - min(px, 12)) ** 2 + (ys - min(py, 12)) ** 2)
            px, py = max(px - 12, 0) + xs[k], max(py - 12, 0) + ys[k]
        cv2.circle(markers, (px, py), 3, i, -1)
    ws = watershed(-dt, markers, mask=reg)
    out = []
    for i in range(1, len(seeds_pt) + 1):
        sub = np.zeros(lab.shape, np.int32)
        sub[y:y + h, x:x + w][ws == i] = 1
        poly = region_polygon(sub, 1, (x, y, w, h), roi, scale, simplify_pt)
        out.append(poly)
    return out
