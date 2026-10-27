"""Extract vector drawing primitives from a PDF page, in *displayed* (rotation-applied) page coordinates (PDF points)."""
from dataclasses import dataclass
import numpy as np
import pymupdf


@dataclass
class Path:
    pts: np.ndarray          # (N,2) polyline in displayed page coords
    closed: bool
    width: float             # stroke width in pt (0 if fill only)
    stroke: tuple | None
    fill: tuple | None
    dashed: bool = False
    has_curve: bool = False


def _bez(p0, p1, p2, p3, n=6):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def extract_paths_reference(page: pymupdf.Page) -> list[Path]:
    """Original (slow, per-point pymupdf.Point) extractor. Kept as the reference that the fast extractor is tested against."""
    rot = page.rotation_matrix
    T = lambda p: np.array(tuple(pymupdf.Point(p) * rot))
    out = []
    for d in page.get_drawings():
        poly, cur, curve = [], None, False
        dashed = d.get('dashes', '[] 0') not in ('[] 0', '', None)

        def flush(closed=False):
            nonlocal poly
            if len(poly) >= 2:
                out.append(Path(np.array(poly), closed or d.get("closePath", False),
                                float(d["width"] or 0) if d["color"] else 0.0, d["color"], d["fill"], dashed, curve))
            poly = []

        for it in d["items"]:
            k = it[0]
            if k == "l":
                a, b = T(it[1]), T(it[2])
                if cur is None or np.linalg.norm(cur - a) > 1e-3:
                    flush(); poly = [a]
                poly.append(b); cur = b
            elif k == "c":
                a, b, c, e = (T(x) for x in it[1:5])
                if cur is None or np.linalg.norm(cur - a) > 1e-3:
                    flush(); poly = [a]
                poly.extend(list(_bez(a, b, c, e)[1:])); cur = e; curve = True
            elif k in ("re", "qu"):
                flush()
                q = pymupdf.Quad(it[1]) if k == "qu" else pymupdf.Rect(it[1]).quad
                pts = [T(p) for p in (q.ul, q.ur, q.lr, q.ll)]
                out.append(Path(np.array(pts), True, float(d["width"] or 0) if d["color"] else 0.0, d["color"], d["fill"], dashed))
                cur = None
        flush()
    return out


def extract_paths(page: pymupdf.Page) -> list[Path]:
    """Fast extractor with output identical to `extract_paths_reference` (same paths, order, flags; coordinates equal to ~1e-9).
    Uses the C-level `get_cdrawings()` (plain tuples) and vectorises all line-only drawings with NumPy: consecutive line items are
    merged into polylines wherever the end of one equals the start of the next (tolerance 1e-3), exactly like the reference.
    Drawings with curves / rectangles / quads take a small per-drawing path. One bulk rotation at the end."""
    cd = page.get_cdrawings()
    rm = page.rotation_matrix
    M = np.array([[rm.a, rm.b], [rm.c, rm.d]], float)
    off = np.array([rm.e, rm.f], float)

    def T(a):
        return np.asarray(a, float) @ M + off

    # ---- split drawings into line-only (fast) and mixed (slow) --------------------------------------------------------
    S, E, DI, mixed = [], [], [], []
    for di, d in enumerate(cd):
        items = d["items"]
        if all(it[0] == "l" for it in items):
            for it in items:
                S.append(it[1]); E.append(it[2]); DI.append(di)
        else:
            mixed.append(di)
    out = {}                                           # drawing index -> list of Path (so the final order is the drawing order)
    meta = lambda d: (float(d.get("width") or 0) if d.get("color") else 0.0, d.get("color"), d.get("fill"), d.get("dashes", "[] 0") not in ("[] 0", "", None))
    if S:
        S = T(np.array(S, float)); E = T(np.array(E, float)); DI = np.array(DI)
        n = len(DI)
        brk = np.ones(n, bool)                         # True where a new polyline starts
        if n > 1:
            cont = (DI[1:] == DI[:-1]) & (np.hypot(*(E[:-1] - S[1:]).T) <= 1e-3)
            brk[1:] = ~cont
        first = np.flatnonzero(brk)
        run_id = np.cumsum(brk) - 1
        n_runs = len(first)
        PTS = np.empty((n + n_runs, 2))
        PTS[first + np.arange(n_runs)] = S[first]                                   # start point of every run
        PTS[np.arange(n) + run_id + 1] = E                                          # every segment's end point
        starts = first + np.arange(n_runs)
        ends = np.append(starts[1:], n + n_runs)
        run_di = DI[first]
        for r in range(n_runs):
            di = int(run_di[r])
            d = cd[di]
            w, col, fil, dash = meta(d)
            out.setdefault(di, []).append(Path(PTS[starts[r]:ends[r]], bool(d.get("closePath", False)), w, col, fil, dash, False))
    # ---- mixed drawings: same logic as the reference, on cdrawings tuples ---------------------------------------------------
    for di in mixed:
        d = cd[di]
        w, col, fil, dash = meta(d)
        res, poly, cur, curve = [], [], None, False

        def flush():
            nonlocal poly
            if len(poly) >= 2:
                res.append(Path(np.array(poly), bool(d.get("closePath", False)), w, col, fil, dash, curve))
            poly = []
        for it in d["items"]:
            k = it[0]
            if k == "l":
                a, b = T(it[1]), T(it[2])
                if cur is None or np.linalg.norm(cur - a) > 1e-3:
                    flush(); poly = [a]
                poly.append(b); cur = b
            elif k == "c":
                a, b, c, e = (T(x) for x in it[1:5])
                if cur is None or np.linalg.norm(cur - a) > 1e-3:
                    flush(); poly = [a]
                poly.extend(list(_bez(a, b, c, e)[1:])); cur = e; curve = True
            elif k in ("re", "qu"):
                flush()
                if k == "re":
                    x0, y0, x1, y1 = it[1]
                    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                else:
                    ul, ur, ll, lr = it[1]
                    corners = [ul, ur, lr, ll]
                res.append(Path(np.array([T(p) for p in corners]), True, w, col, fil, dash))
                cur = None
        flush()
        out[di] = res
    paths = []
    for di in sorted(out):
        paths.extend(out[di])
    return paths
