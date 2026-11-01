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


def find_dash_chains(paths, min_w, pt_per_m, min_chain=6, min_gap_pt=1.0, max_gap_m=6.0, gap_share=0.6, gap_tol=0.5, ang_tol_deg=0.5, off_tol_pt=0.5):
    """Indices of dash / dash-dot lines at ANY angle and line weight (sight lines, axes, grid and centre lines): runs of >= min_chain collinear
    single segments whose gaps are visible (>= min_gap_pt), shorter than max_gap_m and REGULAR - at least gap_share of the gaps lie within gap_tol (a fraction) of
    the median gap (a line type repeats its gap; dash-dot repeats two alternating gaps; an occasional cut by other geometry is tolerated).
    A wall drawn as several segments has touching ends, and the door / window openings along a wall leave gaps of irregular width, so they
    are not matched. Generalises find_grid_paths (axis-aligned, one weight)."""
    idx = [i for i, q in enumerate(paths) if q.width >= min_w and len(q.pts) == 2 and not q.dashed]
    if len(idx) < min_chain:
        return set()
    A = np.array([paths[i].pts[0] for i in idx])
    B = np.array([paths[i].pts[1] for i in idx])
    d = B - A
    L = np.hypot(d[:, 0], d[:, 1])
    ok = L > 0.5
    idx, A, B, d, L = np.array(idx)[ok], A[ok], B[ok], d[ok], L[ok]
    th = np.arctan2(d[:, 1], d[:, 0]) % np.pi
    out = set()
    max_gap = max_gap_m * pt_per_m

    def regular(ga):
        """Most gaps equal the median gap (a line type repeats its gap; an occasional cut by other geometry is tolerated)."""
        if len(ga) < 3:
            return False
        med = float(np.median(ga))
        return med >= min_gap_pt and float(np.mean(np.abs(ga - med) <= gap_tol * med)) >= gap_share

    def accept(chain, gaps):
        if len(chain) < min_chain or not gaps:
            return False
        ga = np.array(gaps)
        if float(np.mean(ga >= min_gap_pt)) < gap_share:
            return False
        return regular(ga) or (regular(ga[0::2]) and regular(ga[1::2]))       # dash line, or dash-dot with two alternating gaps

    for shift in (0.0, np.pi / 2):                       # a second pass with rotated angles so that no group is split at the 0 / pi wrap
        ths = (th + shift) % np.pi
        order = np.argsort(ths)
        grp, start = [], 0
        for a in range(1, len(order) + 1):
            if a == len(order) or ths[order[a]] - ths[order[a - 1]] > np.deg2rad(ang_tol_deg):
                grp.append(order[start:a])
                start = a
        for g in grp:
            if len(g) < min_chain:
                continue
            u = np.c_[np.cos(th[g]), np.sin(th[g])]
            n = np.c_[-np.sin(th[g]), np.cos(th[g])]
            rho = (A[g] * n).sum(1)
            t0, t1 = (A[g] * u).sum(1), (B[g] * u).sum(1)
            lo, hi = np.minimum(t0, t1), np.maximum(t0, t1)
            o = np.argsort(rho)
            s0 = 0
            for a in range(1, len(o) + 1):
                if a == len(o) or rho[o[a]] - rho[o[a - 1]] > off_tol_pt:
                    sub = o[s0:a]
                    s0 = a
                    if len(sub) < min_chain:
                        continue
                    sub = sub[np.argsort(lo[sub])]
                    chain = [sub[0]]
                    reach = hi[sub[0]]
                    gaps = []
                    for j in sub[1:]:
                        gap = lo[j] - reach
                        if gap <= max_gap:
                            gaps.append(gap)
                            chain.append(j)
                            reach = max(reach, hi[j])
                        else:
                            if accept(chain, gaps):
                                out.update(int(idx[g[c]]) for c in chain)
                            chain, reach, gaps = [j], hi[j], []
                    if accept(chain, gaps):
                        out.update(int(idx[g[c]]) for c in chain)
    return out


def find_long_thin_lines(paths, exclude, w_lo, w_hi, pt_per_m, region, min_len_m=4.0, straight_tol_pt=0.6, series_min=4, series_gap_m=1.2):
    """Indices of long straight strokes drawn in the CLUTTER weight (w_lo <= width < w_hi) inside `region`: glazed screens and partitions that CAD
    drawings give the same light weight as furniture. A line carries structure only when it is long (>= min_len_m) and straight; furniture,
    hatching and stair treads are short, or come in regular parallel series (>= series_min lines at <= series_gap_m spacing, overlapping), which are dropped."""
    from shapely.geometry import Point
    cand = []
    for i, q in enumerate(paths):
        if i in exclude or q.width < w_lo or q.width >= w_hi or len(q.pts) < 2:
            continue
        if float(np.hypot(*(q.pts[0] - q.pts[-1]))) < 1e-6:
            continue                                  # closed / back-and-forth paths are outlines and pattern artefacts, not screens (measured: including them lowered agreement)
        c = q.pts.mean(0)
        u = np.linalg.svd(q.pts - c, full_matrices=False)[2][0]
        proj = (q.pts - c) @ u
        L = float(proj.max() - proj.min())
        if L < min_len_m * pt_per_m:
            continue
        if float(np.abs((q.pts - c) @ np.array([-u[1], u[0]])).max()) > straight_tol_pt:
            continue
        a, b = c + u * proj.min(), c + u * proj.max()
        if not region.contains(Point(c)):
            continue
        cand.append((i, a, b, L))
    if not cand:
        return set()
    ang = np.array([np.arctan2(b[1] - a[1], b[0] - a[0]) % np.pi for _, a, b, _ in cand])
    rho = np.array([float(np.dot(a, [-np.sin(t), np.cos(t)])) for (_, a, _, _), t in zip(cand, ang)])
    tt = np.array([[float(np.dot(a, [np.cos(t), np.sin(t)])), float(np.dot(b, [np.cos(t), np.sin(t)]))] for (_, a, b, _), t in zip(cand, ang)])
    drop = set()
    order = np.argsort(ang)
    grp, start = [], 0
    for k in range(1, len(order) + 1):
        if k == len(order) or ang[order[k]] - ang[order[k - 1]] > np.deg2rad(2.0):
            grp.append(order[start:k])
            start = k
    for g in grp:
        if len(g) < series_min:
            continue
        g = g[np.argsort(rho[g])]
        run = [g[0]]
        for j in list(g[1:]) + [None]:
            if j is not None and 0.02 * pt_per_m < rho[j] - rho[run[-1]] <= series_gap_m * pt_per_m and                min(tt[j].max(), tt[run[-1]].max()) - max(tt[j].min(), tt[run[-1]].min()) > 0.5 * min(cand[j][3], cand[run[-1]][3]):
                run.append(j)
                continue
            if len(run) >= series_min:
                drop.update(cand[c][0] for c in run)
            run = [j] if j is not None else []
    return {c[0] for c in cand} - drop


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
