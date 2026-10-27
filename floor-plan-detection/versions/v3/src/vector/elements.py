"""Architectural element detection from vector paths: doors, windows, columns, sanitary-fixture evidence."""
import numpy as np
from shapely.geometry import Polygon, LineString, box, Point
from shapely.ops import unary_union
from .doors import fit_circle


def door_elements(arcs, chords):
    out = []
    for a, (c, e, _) in zip(arcs, chords):
        ang0 = np.degrees(np.arctan2(*(a.p_start - a.center)[::-1])); ang1 = np.degrees(np.arctan2(*(a.p_end - a.center)[::-1]))
        out.append(dict(type="door", hinge_pt=a.center.round(2).tolist(), leaf_radius_pt=round(float(a.radius), 2),
                        swing_sweep_deg=round(float(a.sweep_deg), 1),
                        opening_pt=[c.round(2).tolist(), e.round(2).tolist()],
                        swing_arc_ends_pt=[a.p_start.round(2).tolist(), a.p_end.round(2).tolist()],
                        source="vector:arc_fit"))
    return out


def column_elements(paths, footprint):
    """Columns = compact, near-square unions of grey-filled triangles (the PDF triangulates filled squares)."""
    tris = [Polygon(q.pts).buffer(0.3) for q in paths
            if q.width == 0 and q.fill is not None and 0.5 <= q.fill[0] <= 0.7 and len(q.pts) >= 3 and Polygon(q.pts).is_valid]
    out = []
    merged = unary_union(tris) if tris else None
    for g in (getattr(merged, "geoms", [merged]) if merged is not None else []):
        minx, miny, maxx, maxy = g.bounds
        w, h = maxx - minx, maxy - miny
        if 8 <= min(w, h) and max(w, h) <= 24 and max(w, h) / min(w, h) < 1.4 and g.area / (w * h) > 0.75                 and footprint.buffer(10).contains(g.representative_point()):
            out.append(dict(type="column", bbox_pt=[round(minx, 2), round(miny, 2), round(maxx, 2), round(maxy, 2)], source="vector:grey_fill"))
    return out


def window_elements(paths, footprint, band_pt=14.0, min_len=10.0, w=0.48):
    """Windows = thin 0.48pt strokes (lines or thin rectangles, axis-aligned) lying in the external-wall band; merged by proximity."""
    ring = footprint.boundary.buffer(band_pt)
    segs = []
    for q in paths:
        if abs(q.width - w) > 0.03 or len(q.pts) > 6:
            continue
        lo, hi = q.pts.min(0), q.pts.max(0)
        ext = hi - lo
        if ext.max() < min_len or ext.min() > 6.0:
            continue
        g = box(*lo, *hi).buffer(1.0)
        if ring.contains(g):
            segs.append(g)
    out = []
    merged = unary_union(segs) if segs else None
    for g in (getattr(merged, "geoms", [merged]) if merged is not None else []):
        if g.area < 60:
            continue
        mrr = g.minimum_rotated_rectangle
        out.append(dict(type="window", polygon_pt=np.array(mrr.exterior.coords)[:-1].round(2).tolist(), source="vector:stroke_band"))
    return out


def circle_fixtures(paths, r_range=(3.0, 25.0), max_res=0.35):
    """Round fixture outlines (basin / WC bowl / shower tray) among the thin 0.12pt furniture strokes. Evidence only."""
    out = []
    for q in paths:
        if not (0.05 <= q.width <= 0.15) or len(q.pts) < 12:
            continue
        c, r, res = fit_circle(q.pts)
        if r_range[0] <= r <= r_range[1] and res < max_res * max(1.0, r / 10):
            a = np.unwrap(np.arctan2(q.pts[:, 1] - c[1], q.pts[:, 0] - c[0]))
            if abs(a[-1] - a[0]) > np.radians(300):
                out.append((c, r))
    return out


def window_elements_template(layer, roi, scale, exemplars, search_boxes, footprint, thresh=0.75, band_pt=16.0, min_ink=0.8, device="cpu"):
    """Windows by template matching of repeated CAD window blocks (0.48pt layer), kept only inside the external-wall band."""
    from .fixtures_template import find_symbol
    ring = footprint.boundary.buffer(band_pt)
    found = []
    for ex in exemplars:
        for m in find_symbol(layer, roi, scale, ex, search_boxes, thresh=thresh, min_dist_pt=10, min_ink=min_ink, device=device):
            c = np.array(m["center_pt"])
            if ring.contains(Point(c)) and all(np.linalg.norm(c - np.array(f["center_pt"])) > 10 for f in found):
                found.append(m)
    out = []
    for m in found:
        b = m["box_pt"]
        out.append(dict(type="window", polygon_pt=[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]],
                        match_score=m["score"], source="vector:symbol_template"))
    for o in out:
        o["polygon_pt"] = [[round(x, 2), round(y, 2)] for x, y in o["polygon_pt"]]
    return out
