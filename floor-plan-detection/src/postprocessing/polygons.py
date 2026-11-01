"""Region label image -> validated shapely polygons in page points."""
import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.validation import make_valid


def largest_polygon(geom):
    """The largest Polygon inside any (nested) geometry, or None."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    parts = [largest_polygon(g) for g in getattr(geom, "geoms", [])]
    parts = [g for g in parts if g is not None]
    return max(parts, key=lambda g: g.area, default=None)


def region_polygon(lab, label, bbox, roi, scale, simplify_pt=0.4, min_hole_pt2=20.0):
    x, y, w, h = bbox
    sub = np.pad((lab[y:y + h, x:x + w] == label).astype(np.uint8), 1)
    cnts, hier = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    to_pt = lambda c: (c[:, 0, :].astype(float) - 1 + [x, y] + 0.5) / scale + roi[:2]
    outers = [i for i, hh in enumerate(hier[0]) if hh[3] == -1]
    big = max(outers, key=lambda i: cv2.contourArea(cnts[i]))
    holes = [to_pt(cnts[i]) for i, hh in enumerate(hier[0]) if hh[3] == big and cv2.contourArea(cnts[i]) / scale ** 2 >= min_hole_pt2]
    poly = Polygon(to_pt(cnts[big]), holes)
    if not poly.is_valid:
        # make_valid returns a GeometryCollection that can hold a MultiPolygon (+ line artefacts) when the contour pinches at pixel joints: flatten
        # recursively, otherwise the region - often the largest one - is silently lost
        poly = largest_polygon(make_valid(poly))
    if poly is None or poly.is_empty:
        return None
    return poly.simplify(simplify_pt, preserve_topology=True)


def polygon_to_record(poly, scale):
    ext = np.array(poly.exterior.coords)[:-1]
    return dict(
        polygon_pt=ext.round(2).tolist(),
        polygon=(ext * scale).round(1).tolist(),
        holes=[(np.array(r.coords)[:-1] * scale).round(1).tolist() for r in poly.interiors],
        area_pixels=round(poly.area * scale ** 2, 1),
    )
