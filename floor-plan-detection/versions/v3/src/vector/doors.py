"""Door-swing arc detection from polylines (arcs are stored as short line segments in this PDF, and a swing is often split
into several path fragments, so fragments sharing a centre and radius are merged before the sweep test)."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DoorArc:
    center: np.ndarray      # hinge (pt)
    radius: float
    p_start: np.ndarray
    p_end: np.ndarray
    sweep_deg: float
    path_index: int
    width: float = 0.0
    path_indices: list = field(default_factory=list)
    n_fragments: int = 1


def fit_circle(pts):
    A = np.c_[2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))]
    b = (pts ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    r = np.sqrt(max(sol[2] + c @ c, 0))
    res = np.abs(np.linalg.norm(pts - c, axis=1) - r).max()
    return c, r, res


def detect_door_arcs(paths, min_w=0.1, r_range=(12.0, 40.0), sweep_range=(60, 120), max_res=0.35, min_pts=4,
                     min_fragment_sweep=10.0, merge_tol=1.5):
    # 1. circular fragments
    frags = []
    for i, q in enumerate(paths):
        if q.closed or q.width < min_w or len(q.pts) < min_pts:
            continue
        c, r, res = fit_circle(q.pts)
        if not (r_range[0] <= r <= r_range[1]) or res > max_res * max(1.0, r / 25):
            continue
        a = np.unwrap(np.arctan2(q.pts[:, 1] - c[1], q.pts[:, 0] - c[0]))
        if abs(np.degrees(a[-1] - a[0])) < min_fragment_sweep:
            continue
        frags.append(dict(i=i, c=c, r=r, w=q.width, pts=q.pts))
    # 2. merge fragments with the same centre / radius (greedy clustering)
    clusters = []
    for f in frags:
        for cl in clusters:
            if np.linalg.norm(cl["c"] - f["c"]) <= merge_tol and abs(cl["r"] - f["r"]) <= merge_tol:
                cl["items"].append(f)
                break
        else:
            clusters.append(dict(c=f["c"], r=f["r"], items=[f]))
    # 3. sweep test on the merged cluster
    out = []
    for cl in clusters:
        pts = np.vstack([f["pts"] for f in cl["items"]])
        c = np.mean([f["c"] for f in cl["items"]], axis=0)
        r = float(np.mean([f["r"] for f in cl["items"]]))
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        ref = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())            # circular mean as unwrap reference
        rel = np.arctan2(np.sin(ang - ref), np.cos(ang - ref))
        sweep = np.degrees(rel.max() - rel.min())
        if not (sweep_range[0] <= sweep <= sweep_range[1]):
            continue
        out.append(DoorArc(c, r, pts[rel.argmin()], pts[rel.argmax()], float(sweep), cl["items"][0]["i"],
                           float(max(f["w"] for f in cl["items"])), [f["i"] for f in cl["items"]], len(cl["items"])))
    return out
