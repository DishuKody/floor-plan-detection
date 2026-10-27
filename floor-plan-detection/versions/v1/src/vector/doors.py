"""Door-swing arc detection from polylines (arcs are stored as short line segments in this PDF)."""
from dataclasses import dataclass
import numpy as np


@dataclass
class DoorArc:
    center: np.ndarray      # hinge (pt)
    radius: float
    p_start: np.ndarray
    p_end: np.ndarray
    sweep_deg: float
    path_index: int


def fit_circle(pts):
    A = np.c_[2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))]
    b = (pts ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    r = np.sqrt(max(sol[2] + c @ c, 0))
    res = np.abs(np.linalg.norm(pts - c, axis=1) - r).max()
    return c, r, res


def detect_door_arcs(paths, min_w=0.2, r_range=(12.0, 40.0), sweep_range=(60, 120), max_res=0.35, min_pts=6):
    out = []
    for i, q in enumerate(paths):
        if q.closed or q.width < min_w or len(q.pts) < min_pts:
            continue
        c, r, res = fit_circle(q.pts)
        if not (r_range[0] <= r <= r_range[1]) or res > max_res * max(1.0, r / 25):
            continue
        a = np.unwrap(np.arctan2(q.pts[:, 1] - c[1], q.pts[:, 0] - c[0]))
        sweep = abs(np.degrees(a[-1] - a[0]))
        if not (sweep_range[0] <= sweep <= sweep_range[1]):
            continue
        out.append(DoorArc(c, r, q.pts[0], q.pts[-1], sweep, i))
    return out
