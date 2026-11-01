"""Open-sided rooms. Rooms along a corridor or concourse are often drawn as an outer wall plus partitions whose ends stop at an imaginary frontage line:
nothing is drawn on that line, yet it is the room's fourth side. The frontage is recovered from the geometry that IS drawn: the tips of partitions,
stubs and door frames. Two tips are joined when closing the link turns a region that was OPEN into a bounded pocket of room size. Links are tried
shortest first on the updated mask, so neighbouring partition ends are joined before longer spans, and a link never cuts a room that is already
closed (the region must be open before the link). All lengths are metres at the printed scale."""
import numpy as np
import cv2
from scipy.spatial import cKDTree
from ..raster.segmentation import wall_endpoints


def _touches(stats_row, shape):
    x, y, w, h = stats_row[:4]
    return x == 0 or y == 0 or x + w >= shape[1] or y + h >= shape[0]


def pocket_area_m2(walls, p0, p1, px_per_m, depth_m, area_m2, ds_px_per_m=10.0, min_solidity=0.7, max_aspect=5.0):
    """Area (m2) of the room-sized pocket that the link p0-p1 closes off an OPEN region, or 0.
    Local window, downsampled; a component touching the window border counts as open (it continues beyond the window)."""
    q = ds_px_per_m / px_per_m
    pad = depth_m * px_per_m
    x0 = int(max(min(p0[0], p1[0]) - pad, 0))
    x1 = int(min(max(p0[0], p1[0]) + pad, walls.shape[1]))
    y0 = int(max(min(p0[1], p1[1]) - pad, 0))
    y1 = int(min(max(p0[1], p1[1]) + pad, walls.shape[0]))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    w2 = cv2.resize((walls[y0:y1, x0:x1] > 0).astype(np.uint8) * 255, None, fx=q, fy=q, interpolation=cv2.INTER_AREA)
    w2 = (w2 > 0).astype(np.uint8)
    a = (np.array(p0, float) - [x0, y0]) * q
    b = (np.array(p1, float) - [x0, y0]) * q
    nb, lab_b, st_b, _ = cv2.connectedComponentsWithStats(1 - w2, connectivity=4)
    linked = w2.copy()
    cv2.line(linked, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), 1, 2)
    na, lab_a, st_a, _ = cv2.connectedComponentsWithStats(1 - linked, connectivity=4)
    mid = (a + b) / 2
    d = b - a
    nrm = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9)
    ids = []
    for sg in (1, -1):
        for off in (2, 3, 5, 8):
            s = np.round(mid + sg * off * nrm).astype(int)
            if 0 <= s[0] < linked.shape[1] and 0 <= s[1] < linked.shape[0] and linked[s[1], s[0]] == 0:
                ids.append((int(lab_a[s[1], s[0]]), int(lab_b[s[1], s[0]])))
                break
    if len(ids) < 2:
        return 0.0
    (a1, b1), (a2, b2) = ids
    if b1 != b2 or a1 == a2 or b1 == 0:
        return 0.0                                                   # the two sides were not one region, or the link separates nothing
    if not _touches(st_b[b1], w2.shape):
        return 0.0                                                   # the region was already closed: never cut an existing room
    best = 0.0
    for ida in (a1, a2):
        if _touches(st_a[ida], linked.shape):
            continue                                                 # still open on this side
        am2 = float(st_a[ida][4]) / ds_px_per_m ** 2
        if not (area_m2[0] <= am2 <= area_m2[1]):
            continue
        cs, _ = cv2.findContours((lab_a == ida).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        hull_a = cv2.contourArea(cv2.convexHull(c))
        if hull_a <= 0 or cv2.contourArea(c) / hull_a < min_solidity:
            continue                                                 # a room pocket is compact; slivers around furniture / columns are not
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if max(rw, rh) / max(min(rw, rh), 1e-6) > max_aspect:
            continue
        if best == 0.0 or am2 < best:
            best = am2
    return best


def close_mouths(walls, k, px_per_m, allowed=None, min_gap_m=1.2, max_gap_m=12.0, area_m2=(4.0, 150.0), depth_m=16.0, min_branch_m=0.6, thickness_px=None,
                 align_tol_deg=35.0, parallel_tol_deg=30.0, collinear_tol_deg=25.0, min_support=1, line_tol_m=0.7, support_ext_m=3.0, debug=None):
    """walls: uint8 wall mask (px). k: wall thickness (px). allowed: optional uint8 mask; a link must lie with its midpoint inside it (building footprint).
    Returns (mask with the mouth links drawn, links [(p0, p1, pocket_m2)], n_tips, n_candidate_pairs)."""
    eps = wall_endpoints(walls, k, rectilinear_only=False, min_branch_k=min_branch_m * px_per_m / k, dir_len_k=max(0.5, min_branch_m) * px_per_m / k)
    if len(eps) < 2:
        return walls, [], len(eps), 0
    E = np.array([e for e, _ in eps])
    D = np.array([d for _, d in eps])
    tree = cKDTree(E)
    ext_px, line_tol_px = support_ext_m * px_per_m, line_tol_m * px_per_m
    pairs = sorted(tree.query_pairs(max_gap_m * px_per_m), key=lambda ij: float(np.hypot(*(E[ij[0]] - E[ij[1]]))))
    out = walls.copy()
    th = max(3, int(thickness_px or k // 2))
    sin_al, cos_par, cos_col = np.sin(np.deg2rad(align_tol_deg)), np.cos(np.deg2rad(parallel_tol_deg)), np.cos(np.deg2rad(collinear_tol_deg))
    links, n_cand = [], 0
    log = [] if debug is not None else None
    for i, j in pairs:
        p0, p1 = E[i], E[j]
        L = float(np.hypot(*(p0 - p1)))
        if L < min_gap_m * px_per_m:
            continue
        why = None
        if allowed is not None:
            m = ((p0 + p1) / 2).astype(int)
            if not (0 <= m[1] < allowed.shape[0] and 0 <= m[0] < allowed.shape[1] and allowed[m[1], m[0]]):
                continue
        u = (p1 - p0) / L
        d0, d1 = D[i], D[j]
        across = abs(float(d0 @ u)) <= sin_al and abs(float(d1 @ u)) <= sin_al and float(d0 @ d1) >= cos_par      # across two roughly parallel partitions
        along = float(d0 @ u) >= cos_col and float(d1 @ -u) >= cos_col                                          # a wall continued to a tip that faces back
        if not (across or along):
            if log is not None: log.append((p0.tolist(), p1.tolist(), 'direction'))
            continue
        # a frontage is a straight RUN of tips (partition ends, stub bars, door frames on one line): another tip must lie on the link's line
        near = tree.query_ball_point((p0 + p1) / 2, L / 2 + ext_px)
        nrm = np.array([-u[1], u[0]])
        support = 0
        for t in near:
            if t == i or t == j:
                continue
            r = E[t] - p0
            if abs(float(r @ nrm)) <= line_tol_px and -ext_px <= float(r @ u) <= L + ext_px:
                support += 1
        if support < min_support:
            if log is not None: log.append((p0.tolist(), p1.tolist(), 'support'))
            continue
        n_cand += 1
        am2 = pocket_area_m2(out, p0, p1, px_per_m, depth_m, area_m2)
        if am2 <= 0 and log is not None:
            log.append((p0.tolist(), p1.tolist(), 'pocket'))
        if am2 > 0:
            cv2.line(out, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), 255, th)
            links.append(((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])), round(am2, 1)))
    if debug is not None:
        debug.update(tips=[(e.tolist(), d.tolist()) for e, d in eps], rejected=log)
    return out, links, len(eps), n_cand
