"""Geometric shape descriptors / classification and axis snapping for room polygons (any units)."""
import numpy as np
import cv2
from shapely.geometry import Polygon


def _min_rect(pts):
    (cx, cy), (w, h), ang = cv2.minAreaRect(np.asarray(pts, np.float32))
    return (cx, cy), max(w, h), min(w, h), ang


def describe(poly: Polygon, axis_tol_deg=3.0):
    """Shape descriptors of the EXTERIOR ring (holes are reported separately). Classification:
    circle | square | rectangle | rectilinear_polygon | irregular_polygon."""
    ext = np.array(poly.exterior.coords)[:-1]
    A, P = poly.area, poly.exterior.length
    (cx, cy), L, S, ang = _min_rect(ext)
    rect_area = max(L * S, 1e-9)
    hull = poly.convex_hull
    solidity = float(Polygon(poly.exterior).area / max(hull.area, 1e-9))
    circ = float(4 * np.pi * Polygon(poly.exterior).area / max(P * P, 1e-9))
    # fraction of boundary length lying on the two dominant (rotated-rect) axes
    ax = np.deg2rad(ang)
    d = np.roll(ext, -1, axis=0) - ext
    ln = np.linalg.norm(d, axis=1)
    th = np.arctan2(d[:, 1], d[:, 0]) - ax
    off = np.abs(((th + np.pi / 4) % (np.pi / 2)) - np.pi / 4)                    # angular offset to the nearest of the 4 axis directions
    ortho = float(ln[off < np.deg2rad(axis_tol_deg)].sum() / max(ln.sum(), 1e-9))
    rectangularity = float(Polygon(poly.exterior).area / rect_area)
    if circ >= 0.88 and rectangularity < 0.9 and solidity >= 0.95:
        kind = "circle"
    elif rectangularity >= 0.95 and solidity >= 0.95 and ortho >= 0.9:
        kind = "square" if L / max(S, 1e-9) <= 1.1 else "rectangle"
    elif ortho >= 0.9:
        kind = "rectilinear_polygon"
    else:
        kind = "irregular_polygon"
    return dict(shape_type=kind, n_vertices=int(len(ext)), n_holes=len(poly.interiors), rectangularity=round(rectangularity, 3),
                solidity=round(solidity, 3), circularity=round(circ, 3), axis_aligned_edge_fraction=round(ortho, 3),
                min_rect_aspect=round(float(L / max(S, 1e-9)), 2), min_rect_angle_deg=round(float(ang), 1), is_convex=bool(solidity > 0.999))


def snap_axis_aligned(poly: Polygon, tol_deg=3.0, max_area_change=0.02):
    """Make near-horizontal / near-vertical edges exactly axis-aligned (shared vertex coordinates), holes included.
    Falls back to the input if the result is invalid or changes the area by more than max_area_change."""
    def snap_ring(ring):
        pts = np.array(ring.coords)[:-1].astype(float)
        n = len(pts)
        if n < 3:
            return pts
        px, py = list(range(n)), list(range(n))

        def find(p, i):
            while p[i] != i:
                p[i] = p[p[i]]
                i = p[i]
            return i

        def union(p, a, b):
            p[find(p, a)] = find(p, b)
        t = np.tan(np.deg2rad(tol_deg))
        for i in range(n):
            j = (i + 1) % n
            dx, dy = abs(pts[j, 0] - pts[i, 0]), abs(pts[j, 1] - pts[i, 1])
            if dy <= t * dx:
                union(py, i, j)          # horizontal edge: same y
            elif dx <= t * dy:
                union(px, i, j)          # vertical edge: same x
        out = pts.copy()
        for p, c in ((px, 0), (py, 1)):
            groups = {}
            for i in range(n):
                groups.setdefault(find(p, i), []).append(i)
            for idx in groups.values():
                if len(idx) > 1:
                    out[idx, c] = pts[idx, c].mean()
        return out
    try:
        q = Polygon(snap_ring(poly.exterior), [snap_ring(r) for r in poly.interiors])
        if not q.is_valid:
            q = q.buffer(0)
        if q.geom_type != "Polygon" or q.is_empty or abs(q.area - poly.area) > max_area_change * poly.area:
            return poly
        return q
    except Exception:
        return poly
