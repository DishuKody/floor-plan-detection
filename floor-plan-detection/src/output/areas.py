"""Independent-area output: every detected area gets a unique ID and a distinct fill colour on the full-page render (the drawing stays visible
underneath), plus a JSON with the area geometry. Colours are assigned by greedy graph colouring of the adjacency (touching areas never share a colour)
from a 12-colour palette, so neighbours are always distinguishable; the ID is printed at the area's pole of inaccessibility."""
import json
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.ops import polylabel

def _make_palette(n=60):
    """n well separated colours (RGB): golden-ratio hues x three saturation/value levels."""
    import colorsys
    out = []
    for i in range(n):
        h = (i * 0.61803398875) % 1.0
        sat, val = ((0.75, 0.95), (0.5, 0.85), (0.9, 0.75))[i % 3]
        out.append(tuple(int(round(255 * c)) for c in colorsys.hsv_to_rgb(h, sat, val)))
    return out


PALETTE = _make_palette(60)


def assign_colours(polys, tol=1.5):
    """Neighbouring areas never share a colour; among the allowed colours the least used one is taken, so colours repeat as rarely as possible."""
    tree = STRtree(polys)
    col = [-1] * len(polys)
    use = [0] * len(PALETTE)
    for i in sorted(range(len(polys)), key=lambda i: -polys[i].area):
        g = polys[i].buffer(tol)
        used = {col[j] for j in tree.query(g) if j != i and col[j] >= 0 and g.intersects(polys[j])}
        c = min((c for c in range(len(PALETTE)) if c not in used), key=lambda c: (use[c], c), default=i % len(PALETTE))
        col[i] = c
        use[c] += 1
    return col


def write_areas(base_bgr, rooms, scale, out_dir, alpha=0.5, min_label_area_pt2=150.0):
    """base_bgr: page render at `scale` px/pt. rooms: room records (polygon_pt, holes, id, type, approx_area_m2). Writes areas_overlay.png and areas.json."""
    polys = []
    for r in rooms:
        ring = np.array(r["polygon_pt"])
        holes = [np.array(h) / scale for h in r.get("holes", [])]
        pg = Polygon(ring, holes).buffer(0)
        polys.append(pg if not pg.is_empty else Polygon(ring))
    cols = assign_colours(polys) if polys else []
    ov = base_bgr.copy()
    for r, pg, c in zip(rooms, polys, cols):
        bgr = tuple(int(v) for v in PALETTE[c][::-1])
        cv2.fillPoly(ov, [np.round(np.array(r["polygon"])).astype(np.int32)], bgr)
        for h in r.get("holes", []):
            cv2.fillPoly(ov, [np.round(np.array(h)).astype(np.int32)], (255, 255, 255))
    img = cv2.addWeighted(base_bgr, 1 - alpha, ov, alpha, 0)
    for r in rooms:
        cv2.polylines(img, [np.round(np.array(r["polygon"])).astype(np.int32)], True, (255, 255, 255), 5, cv2.LINE_AA)      # clear boundary between neighbours
        cv2.polylines(img, [np.round(np.array(r["polygon"])).astype(np.int32)], True, (40, 40, 40), 2, cv2.LINE_AA)
    items = []
    for r, pg, c in zip(rooms, polys, cols):
        pole = polylabel(pg, 0.5) if pg.geom_type == "Polygon" else pg.representative_point()
        if pg.area >= min_label_area_pt2:
            txt = r["id"].replace("room-", "")
            h = max(0.55, min(1.4, np.sqrt(pg.area) / 70.0))
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, h, 2)
            x, y = int(pole.x * scale - tw / 2), int(pole.y * scale + th / 2)
            cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, h, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, h, (0, 0, 0), 2, cv2.LINE_AA)
        items.append(dict(id=r["id"], colour_rgb=list(PALETTE[c]), type=r["type"], area_m2=r["approx_area_m2"], polygon_pt=r["polygon_pt"], polygon_px=r["polygon"],
                          holes_px=r.get("holes", []), centroid_pt=[round(pg.centroid.x, 2), round(pg.centroid.y, 2)], label_point_pt=[round(pole.x, 2), round(pole.y, 2)],
                          shape=r.get("shape", {}).get("shape_type"), flags=r.get("flags", [])))
    out = Path(out_dir)
    assert img.shape == base_bgr.shape
    cv2.imwrite(str(out / "areas_overlay.png"), img)
    json.dump(dict(n_areas=len(items), scale_px_per_pt=scale, page_px=[base_bgr.shape[1], base_bgr.shape[0]], areas=items), open(out / "areas.json", "w"))
    return len(items)
