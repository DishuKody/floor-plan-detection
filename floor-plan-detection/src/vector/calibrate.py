"""Sheet calibration for vector plans: nothing here is tuned to one PDF; every value is derived from the drawing itself
(or from the explicitly configured printed scale)."""
from collections import defaultdict
import numpy as np
import cv2
from shapely.geometry import box, Point, Polygon
from shapely.ops import unary_union


def width_classes(paths):
    """Total stroke length per rounded line weight (pt) of line-work (stroked, not filled)."""
    L = defaultdict(float)
    for q in paths:
        if q.width > 0:
            L[round(q.width, 2)] += float(np.linalg.norm(np.diff(q.pts, axis=0), axis=1).sum())
    return dict(L)


def structural_stroke(paths, factor=1.5):
    """Wall/structure line weight threshold relative to the sheet's dominant (clutter) line weight: the weight carrying the most
    ink is furniture / hatching / annotation; structure is drawn clearly heavier. factor 1.5 reproduces 0.2 pt on a sheet whose
    dominant weight is 0.12 pt and gives 0.32 pt where it is 0.21 pt."""
    L = width_classes(paths)
    if not L:
        return 0.0, None, L
    dom = max(L, key=L.get)
    return round(dom * factor, 3), dom, L


def plan_footprint(paths, page_rect, min_w, scale_ratio, max_line_frac=0.3, close_m=3.0, min_ink_frac=0.1, min_paths=40, neck_m=10.0, isolated_frac=0.08, open_m=2.0, px_per_pt=0.5):
    """Building footprint polygon (page pt) from the structural line layer.
    Long, nearly straight lines (sheet frame, title-block cells, site boundaries, grid axes) are not building walls and are dropped;
    the rest is closed morphologically at `close_m` metres (so a building's walls fuse into one cluster) and clusters carrying at least
    `min_ink_frac` of the heaviest cluster's ink are kept. Returns (polygon | None, info)."""
    W, H = page_rect.width, page_rect.height
    s = px_per_pt
    m = np.zeros((int(H * s) + 1, int(W * s) + 1), np.uint8)
    max_len = max_line_frac * max(W, H)
    ink = 0.0
    used = []
    used_pt = []
    cand = [q for q in paths if q.width >= min_w and not q.dashed]
    mids = np.array([(q.pts.min(0) + q.pts.max(0)) / 2 for q in cand]) if cand else np.zeros((0, 2))
    iso_len = isolated_frac * max(W, H)
    for q in cand:
        seg = np.linalg.norm(np.diff(q.pts, axis=0), axis=1)
        ext = float(np.ptp(q.pts, axis=0).max())
        if ext > max_len:
            continue                                  # long straight line: frame / site boundary / title block / axis
        if ext > iso_len:
            lo, hi = q.pts.min(0), q.pts.max(0)
            near = mids[(mids > lo).all(1) & (mids < hi).all(1)]
            cnt32 = q.pts.astype(np.float32).reshape(-1, 1, 2)
            inside = sum(cv2.pointPolygonTest(cnt32, (float(a), float(b)), False) > 0 for a, b in near)
            if inside < 10:
                continue                              # long outline enclosing (almost) no structure: pond, planting bed, site feature
        cv2.polylines(m, [(q.pts * s).astype(np.int32)], q.closed, 255, 1)
        ink += float(seg.sum())
        used.append(((q.pts[0] + q.pts[-1]) / 2 * s).astype(int))
        used_pt.append(tuple((q.pts.min(0) + q.pts.max(0)) / 2))
    if ink == 0:
        return None, dict(reason="no structural ink")
    pt_per_m = 1000.0 / (25.4 / 72 * scale_ratio)
    k = max(3, int(close_m * pt_per_m * s) | 1)
    mc = cv2.morphologyEx(cv2.copyMakeBorder(m, k, k, k, k, cv2.BORDER_CONSTANT), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))[k:-k, k:-k]
    n, lab, st, _ = cv2.connectedComponentsWithStats(mc, connectivity=8)
    ink_px = {i: int(((lab == i) & (m > 0)).sum()) for i in range(1, n)}
    cnt = defaultdict(int)
    for c in used:
        cnt[int(lab[min(max(c[1], 0), lab.shape[0] - 1), min(max(c[0], 0), lab.shape[1] - 1)])] += 1
    # a real building is many paths (walls, doors, stairs); a lone pond outline, key-plan outline or stamp box is not
    keep = [i for i, v in ink_px.items() if v >= min_ink_frac * max(ink_px.values()) and cnt.get(i, 0) >= min_paths]     # clusters carrying real structure (not a pond outline / title-block table)
    polys = []
    for i in keep:
        cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if len(c) >= 3:
                polys.append(Polygon(c[:, 0, :] / s).buffer(0))
    if not polys:
        return None, dict(reason="no dense structural cluster")
    fp = unary_union(polys).buffer(0)
    # morphological closing pulls the outline half a kernel away from the walls: undo that part so the footprint hugs the building
    fp = fp.buffer(-0.35 * k / s).buffer(0.35 * k / s) if fp.area > 0 else fp
    r_open = open_m * pt_per_m                         # opening removes thin spurs (grid frames, dimension lines) that closing welded on
    fp = fp.buffer(-r_open).buffer(r_open)
    if fp.is_empty:
        return None, dict(reason="footprint vanished after opening")
    # split at necks narrower than `neck_m` and keep only parts that hold structure (a site feature welded to the building by closing is dropped)
    r_neck = neck_m * pt_per_m / 2
    core = fp.buffer(-r_neck).buffer(r_neck)
    parts = [g for g in getattr(core, "geoms", [core]) if g.area > 0]
    if len(parts) > 1:
        pts = [Point(c) for c in used_pt]
        n_in = [sum(g.contains(p) for p in pts) for g in parts]
        good = [g for g, n in zip(parts, n_in) if n >= min_paths and n >= min_ink_frac * max(n_in)]
        if good and len(good) < len(parts):
            fp = fp.intersection(unary_union(good).buffer(1.2 * r_neck))
    return fp, dict(n_clusters=len(keep), close_px=k, structural_ink_pt=round(ink), area_pt2=round(fp.area))
