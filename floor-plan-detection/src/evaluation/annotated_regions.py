"""Room-level comparison with a hand-annotated reference (red boundary lines drawn over a screenshot of the sheet).

The red lines close the open sides of rooms (corridor frontage lines, partitions, envelopes). EXPECTED ROOMS = the enclosed regions of
(structural walls from the vector PDF  UNION  the red lines) that touch a red line. Detected rooms are matched to them by overlap.
The structural walls are rasterised straight from the PDF (not from our own pipeline output), so the expected rooms do not contain our own
virtual boundaries. Door openings are closed with the detected door chords (otherwise every classroom leaks into the corridor).

Per expected room: best IoU with any single detected room (an over-merged detection and an over-split one both score low), and the share of the
expected room covered by its best detected room."""
import json
from pathlib import Path
import cv2
import numpy as np
import pymupdf

from ..vector.extract_pdf_paths import extract_paths
from ..vector.walls import rasterize_mask, find_grid_paths, find_dash_chains
from . import expected_ref as E


def structural_wall_mask(pdf, scale, min_w, pt_per_m, close_pt=1.75):
    """Page-sized uint8 mask (1 = wall) at `scale` px/pt: structural strokes + poche fills, channels between double lines closed.
    Dash / dash-dot lines (axes, grids, boundaries) are not walls."""
    page = pymupdf.open(pdf)[0]
    P = extract_paths(page)
    roi = (0.0, 0.0, float(page.rect.width), float(page.rect.height))
    skip = find_grid_paths(P) | find_dash_chains(P, min_w, pt_per_m) | {i for i, q in enumerate(P) if q.dashed}
    m = rasterize_mask(P, skip, roi, scale, min_w=min_w)
    k = int(close_pt * scale) | 1
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))


def red_line_mask(marks, A, shape, thickness_px=5):
    m = np.zeros(shape[:2], np.uint8)
    for ln in marks["lines"]:
        pts = ln["pts"] @ A[:, :2].T + A[:, 2]
        for x, y in np.round(pts).astype(int):
            cv2.circle(m, (int(x), int(y)), thickness_px // 2, 255, -1)
    return m


def expected_regions(wall_mask, red_mask, door_chords_px, px_per_m, footprint_px=None, area_m2=(2.0, 220.0), touch_px=6, gap_px=7):
    """-> (label image, list of dict(label, area_m2, bbox, centroid)) of valid expected rooms.
    Hand-drawn lines stop a few pixels short of walls: the ink is closed by gap_px for labelling, then every region grows back into the free space
    it lost (so the region boundaries stay on the lines / wall faces)."""
    ink = ((wall_mask > 0) | (red_mask > 0)).astype(np.uint8) * 255
    for a, b in door_chords_px:
        cv2.line(ink, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), 255, 3)
    closed = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap_px + 1,) * 2))
    n, lab, st, cen = cv2.connectedComponentsWithStats((closed == 0).astype(np.uint8), connectivity=4)
    free = ink == 0
    grown = lab.astype(np.int32).copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(gap_px):
        d = cv2.dilate(grown.astype(np.float32), k).astype(np.int32)
        take = free & (grown == 0)
        grown[take] = d[take]
    lab = grown
    near_red = cv2.dilate(red_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * touch_px + 1,) * 2)) > 0
    H, W = ink.shape
    out = []
    areas = np.bincount(lab.ravel(), minlength=n)
    for i in range(1, n):
        am2 = areas[i] / px_per_m ** 2
        if not (area_m2[0] <= am2 <= area_m2[1]):
            continue
        ys, xs = np.nonzero(lab == i)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        if x0 == 0 or y0 == 0 or x1 >= W or y1 >= H:
            continue
        reg = lab[y0:y1, x0:x1] == i
        if not (near_red[y0:y1, x0:x1] & cv2.dilate(reg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).any():
            continue
        c = (float(xs.mean()), float(ys.mean()))
        if footprint_px is not None and cv2.pointPolygonTest(footprint_px, c, False) < 0:
            continue
        out.append(dict(label=i, area_m2=round(float(am2), 1), bbox=[x0, y0, x1, y1], centroid=[round(c[0], 1), round(c[1], 1)]))
    return lab, out


def detected_id_map(rooms, shape):
    """int32 map: 1 + room index inside the room polygon (holes cut out), larger rooms painted first so a small room inside a big one wins."""
    m = np.zeros(shape[:2], np.int32)
    order = sorted(range(len(rooms)), key=lambda i: -abs(cv2.contourArea(np.array(rooms[i]["polygon"], np.float32))))
    for i in order:
        r = rooms[i]
        for poly in [r["polygon"]] + list(r.get("extra_polygons", [])):
            cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], i + 1)
        for h in r.get("holes", []):
            cv2.fillPoly(m, [np.round(np.array(h)).astype(np.int32)], 0)
    return m


def compare(lab, regions, ids, n_rooms):
    """Best IoU of every expected room against any single detected room."""
    areas = np.bincount(ids.ravel(), minlength=n_rooms + 1)
    rows = []
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        reg = lab[y0:y1, x0:x1] == r["label"]
        inter = np.bincount(ids[y0:y1, x0:x1][reg], minlength=n_rooms + 1)
        inter[0] = 0
        a_reg = int(reg.sum())
        iou = inter / np.maximum(areas + a_reg - inter, 1)
        j = int(np.argmax(iou))
        rows.append(dict(r, best_room_index=j - 1 if iou[j] > 0 else None, iou=round(float(iou[j]), 3), covered=round(float(inter[j]) / max(a_reg, 1), 3),
                         detected_area_ratio=round(float(areas[j]) / max(a_reg, 1), 2) if iou[j] > 0 else None))
    return rows


def summarize(rows):
    iou = np.array([r["iou"] for r in rows]) if rows else np.zeros(0)
    return dict(n_expected_rooms=len(rows), mean_iou=round(float(iou.mean()), 3) if len(iou) else None, median_iou=round(float(np.median(iou)), 3) if len(iou) else None,
                share_iou_ge_0_5=round(float((iou >= 0.5).mean()), 3) if len(iou) else None, share_iou_ge_0_7=round(float((iou >= 0.7).mean()), 3) if len(iou) else None,
                share_iou_ge_0_85=round(float((iou >= 0.85).mean()), 3) if len(iou) else None)


def evaluate(pdf, out_dir, annotated_png, scale_hint=3.0, min_w=0.315, px_per_pt=1.5, pt_per_m=14.173):
    """Full pipeline for one output folder. Returns (summary, rows, images dict of BGR arrays)."""
    out_dir = Path(out_dir)
    exp = cv2.imread(str(annotated_png))
    marks = E.extract_marks(exp, networks=True)
    J = json.load(open(out_dir / "rooms.json"))
    rooms = J["rooms"]
    target = cv2.imread(str(out_dir / "debug" / "render_reference.png"))
    A, reg = E.register(exp, target, marks["mask"], scale_hint)
    shape = target.shape
    wall = structural_wall_mask(pdf, px_per_pt, min_w, pt_per_m)
    red = red_line_mask(marks, A, shape)
    els = json.load(open(out_dir / "elements.json"))["elements"]
    chords = [(np.array(e["opening_pt"][0]) * px_per_pt, np.array(e["opening_pt"][1]) * px_per_pt) for e in els if e["type"] == "door"]
    px_per_m = pt_per_m * px_per_pt
    fps = J["image"].get("building_footprint_pt", [])
    foot = None
    if fps:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        u = unary_union([Polygon(p) for p in fps]).buffer(0)
        g = max(getattr(u, "geoms", [u]), key=lambda q: q.area)
        foot = (np.array(g.exterior.coords) * px_per_pt).astype(np.float32).reshape(-1, 1, 2)
    lab, regions = expected_regions(wall, red, chords, px_per_m, foot)
    ids = detected_id_map(rooms, shape)
    rows = compare(lab, regions, ids, len(rooms))
    S = summarize(rows)
    S["registration"] = reg
    return S, rows, dict(lab=lab, ids=ids, red=red, wall=wall, target=target, A=A, marks=marks, exp=exp)
