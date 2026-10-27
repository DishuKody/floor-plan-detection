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


def extract_paths(page: pymupdf.Page) -> list[Path]:
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
