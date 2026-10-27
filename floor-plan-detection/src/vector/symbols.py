"""Symbol linework that must not be treated as walls: shaft 'X' crosses and stair-tread series."""
import numpy as np


def _segments(paths, skip, w_max=0.5, w_min=0.1):
    return [(i, q.pts[0], q.pts[1]) for i, q in enumerate(paths)
            if i not in skip and len(q.pts) == 2 and q.width and w_min <= q.width <= w_max]


def find_cross_symbols(paths, skip=(), min_len=20.0, mid_tol=2.0, angle_range=(20, 70)):
    """Pairs of long diagonal segments that cross at both midpoints (shaft / void 'X'). Returns the path indices."""
    diag = []
    for i, a, b in _segments(paths, skip):
        d = b - a
        L = np.linalg.norm(d)
        ang = abs(np.degrees(np.arctan2(d[1], d[0]))) % 90
        if L >= min_len and angle_range[0] <= ang <= angle_range[1]:
            diag.append((i, (a + b) / 2, L))
    out = set()
    for k, (i, m, L) in enumerate(diag):
        for j, m2, L2 in diag[k + 1:]:
            if np.linalg.norm(m - m2) <= mid_tol and abs(L - L2) <= 2.0:
                out.update((i, j))
    return out


def find_tread_series(paths, skip=(), min_run=5, spacing=(3.5, 14.0), spacing_tol=0.8, len_tol=1.5, len_range=(8.0, 90.0)):
    """Runs of >=min_run parallel, equal-length, equally spaced axis-aligned thin segments (stair treads / hatch)."""
    groups = {}
    for i, a, b in _segments(paths, skip):
        d = b - a
        L = np.linalg.norm(d)
        if not (len_range[0] <= L <= len_range[1]):
            continue
        if abs(d[0]) < 0.05:
            groups.setdefault(("v", round(L / len_tol), round(a[1] / len_tol / 2 if a[1] < b[1] else b[1] / len_tol / 2)), []).append((a[0], i))
        elif abs(d[1]) < 0.05:
            groups.setdefault(("h", round(L / len_tol), round(min(a[0], b[0]) / len_tol / 2)), []).append((a[1], i))
    out = set()
    for items in groups.values():
        items.sort()
        run = [items[0]]
        for it in items[1:]:
            gap = it[0] - run[-1][0]
            if spacing[0] <= gap <= spacing[1] and (len(run) < 2 or abs(gap - (run[-1][0] - run[-2][0])) <= spacing_tol):
                run.append(it)
            else:
                if len(run) >= min_run:
                    out.update(x[1] for x in run)
                run = [it]
        if len(run) >= min_run:
            out.update(x[1] for x in run)
    return out
