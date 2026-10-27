"""Wall-mask construction from classified vector paths."""
import numpy as np
import cv2


def find_grid_paths(paths, w=0.24, tol=0.03, min_chain=4, max_gap=14.0, min_extent=100.0):
    """Indices of dash-dot grid/axis lines: chains of collinear 2-point axis-aligned 0.24pt strokes with regular gaps."""
    groups = {}
    for i, q in enumerate(paths):
        if abs(q.width - w) > tol or len(q.pts) != 2:
            continue
        d = q.pts[1] - q.pts[0]
        if abs(d[0]) < 0.05:
            groups.setdefault(("v", round(q.pts[0, 0], 1)), []).append((q.pts[:, 1].min(), q.pts[:, 1].max(), i))
        elif abs(d[1]) < 0.05:
            groups.setdefault(("h", round(q.pts[0, 1], 1)), []).append((q.pts[:, 0].min(), q.pts[:, 0].max(), i))
    grid = set()
    for segs in groups.values():
        segs.sort()
        chain = [segs[0]]
        def flush():
            if len(chain) >= min_chain and chain[-1][1] - chain[0][0] >= min_extent:
                grid.update(c[2] for c in chain)
        for s in segs[1:]:
            if 0 <= s[0] - chain[-1][1] <= max_gap:
                chain.append(s)
            else:
                flush(); chain = [s]
        flush()
    return grid


def rasterize_mask(paths, skip, roi, scale, min_w=0.2, fill_gray=(0.5, 0.95), min_black_extent=9.0):
    x0, y0, x1, y1 = roi
    W, H = int((x1 - x0) * scale), int((y1 - y0) * scale)
    m = np.zeros((H, W), np.uint8)
    for i, q in enumerate(paths):
        if i in skip:
            continue
        pts = ((q.pts - [x0, y0]) * scale).astype(np.int32)
        if q.width == 0:
            if q.fill is None or len(pts) < 3:
                continue
            ext = float(np.ptp(q.pts, axis=0).max())
            gray = fill_gray[0] <= q.fill[0] <= fill_gray[1]
            # black fills are mostly text glyphs (<~6pt); long black fills are wall poche
            # black fills: wall poche is a 3-5 point rectangle/sliver; text glyphs (>=6 points, even large headings) are not walls
            if gray or (q.fill[0] < 0.5 and ext >= min_black_extent and len(q.pts) <= 5):
                cv2.fillPoly(m, [pts], 255)
        elif q.width >= min_w:
            cv2.polylines(m, [pts], q.closed, 255, 1)
    return m


def _inside(pt, roi, shape, scale):
    xi, yi = int((pt[0] - roi[0]) * scale), int((pt[1] - roi[1]) * scale)
    return 1 <= xi < shape[1] - 1 and 1 <= yi < shape[0] - 1


def _hit(mask, pt, roi, scale):
    xi, yi = int((pt[0] - roi[0]) * scale), int((pt[1] - roi[1]) * scale)
    if not (1 <= xi < mask.shape[1] - 1 and 1 <= yi < mask.shape[0] - 1):
        return False
    return bool(mask[yi - 1:yi + 2, xi - 1:xi + 2].any())


def extend_dangling_walls(paths, skip, mask, roi, scale, min_len=40.0, max_ext=30.0, min_gap=2.5, w_range=(0.2, 1.5),
                          lateral=(4.0, 14.0)):
    """Extend free ends of long axis-aligned wall lines along their axis until they hit another wall (drafting convention:
    partitions drawn short of the wall they meet). Returns the extension segments added (page pt)."""
    x0, y0 = roi[:2]
    H, W = mask.shape
    added = []
    steps = np.arange(min_gap, max_ext, 0.5)
    lat = np.arange(lateral[0], lateral[1], 0.5)
    segs = []
    for i, q in enumerate(paths):
        if i in skip:
            continue
        if len(q.pts) == 2 and w_range[0] <= q.width <= w_range[1]:
            segs.append((q.pts[0], q.pts[1]))
        elif q.width == 0 and q.fill is not None and q.fill[0] < 0.95 and len(q.pts) >= 3:
            lo, hi = q.pts.min(0), q.pts.max(0)
            ext = hi - lo
            if ext.max() >= min_len and ext.min() <= 6.0:   # elongated wall poche rectangle -> centreline
                c = (lo + hi) / 2
                if ext[0] > ext[1]:
                    segs.append((np.array([lo[0], c[1]]), np.array([hi[0], c[1]])))
                else:
                    segs.append((np.array([c[0], lo[1]]), np.array([c[0], hi[1]])))
    for a, b in segs:
        d = b - a
        L = np.linalg.norm(d)
        if L < min_len or (abs(d[0]) > 0.05 and abs(d[1]) > 0.05):
            continue
        u = d / L
        n = np.array([-u[1], u[0]])
        for end, dirn in ((b, u), (a, -u)):
            for s in steps:
                pt = end + dirn * s
                if not _inside(pt, roi, mask.shape, scale):
                    break
                if _hit(mask, pt, roi, scale):
                    if s > min_gap:
                        added.append((end.copy(), pt.copy()))
                    break
                # T-junction through a gap: collinear wall ends on both sides of the ray (e.g. stem pointing into a door-frame gap)
                if s > min_gap and all(any(_hit(mask, pt + sgn * n * t, roi, scale) for t in lat) for sgn in (-1, 1)):
                    added.append((end.copy(), pt.copy()))
                    break
    for p0, p1 in added:
        cv2.line(mask, tuple(int(v) for v in (p0 - [x0, y0]) * scale), tuple(int(v) for v in (p1 - [x0, y0]) * scale), 255, 2)
    return added



def hairline_partitions(paths, skip, mask, roi, scale, min_len=15.0, touch_pt=1.5, w_range=(0.05, 0.2)):
    """Hairline (0.12pt) straight axis-aligned segments that touch existing wall mask at BOTH ends: drawn walls of service rooms /
    door walls that are plotted at furniture weight. Furniture edges rarely touch two walls, so they are not added.
    Returns list of (p0, p1) in page pt; does not modify mask."""
    x0, y0 = roi[:2]
    H, W = mask.shape
    r = max(int(touch_pt * scale), 1)

    def touches(p):
        xi, yi = int((p[0] - x0) * scale), int((p[1] - y0) * scale)
        if not (r <= xi < W - r and r <= yi < H - r):
            return False
        return bool(mask[yi - r:yi + r + 1, xi - r:xi + r + 1].any())

    out = []
    for i, q in enumerate(paths):
        if i in skip or len(q.pts) != 2 or q.width == 0 or not (w_range[0] <= q.width < w_range[1]):
            continue
        a, b = q.pts
        d = b - a
        if np.linalg.norm(d) < min_len or (abs(d[0]) > 0.05 and abs(d[1]) > 0.05):
            continue
        u = d / np.linalg.norm(d)
        if touches(a - u * 0.5) and touches(b + u * 0.5):
            out.append((a.copy(), b.copy()))
    return out


def rasterize_thick(paths, skip, roi, scale, min_stroke=0.9, min_black_extent=9.0, min_thickness=2.0):
    """Mask of structural walls only: heavy strokes (>= min_stroke) and long black poche fills at least min_thickness thick."""
    x0, y0, x1, y1 = roi
    m = np.zeros((int((y1 - y0) * scale), int((x1 - x0) * scale)), np.uint8)
    for i, q in enumerate(paths):
        if i in skip:
            continue
        pts = ((q.pts - [x0, y0]) * scale).astype(np.int32)
        if q.width >= min_stroke:
            cv2.polylines(m, [pts], q.closed, 255, 1)
        elif q.width == 0 and q.fill is not None and q.fill[0] < 0.5 and 3 <= len(pts) <= 5:
            ext = np.ptp(q.pts, axis=0)
            if ext.max() >= min_black_extent and ext.min() >= min_thickness:
                cv2.fillPoly(m, [pts], 255)
    return m
