"""Raster-only floor plans (scanned/exported images embedded in a PDF): classical CV + OCR, no trained room model.

Pipeline: Otsu -> thick-wall mask (closing + thickness opening) -> building footprint -> OCR labels as room seeds ->
marker-controlled watershed on the distance transform (cuts fall at door necks) -> classification from the label text ->
door arcs by circle fitting on thin linework. Polygons are written in BOTH frames: source-image pixels and page-render pixels."""
import re
import time
from pathlib import Path
import numpy as np
import cv2
import pymupdf
from shapely.geometry import Polygon, Point
from skimage.graph import MCP_Geometric

from ..output.json_writer import write_json
from ..output.visualization import draw_overlay, add_legend_inset
from ..postprocessing.polygons import region_polygon
from ..gpu import resolve_device

# ordered: first match wins ("Kitchen veranda" must be a balcony, "Covered Veranda" too)
RULES = [("balcony", ["veranda", "porch", "balcony", "terrace"]), ("toilet_or_wc", ["wc", "w.c", "toilet"]),
         ("bathroom", ["bath", "shower", "ensuite"]), ("kitchen", ["kitchen"]), ("bedroom", ["bedroom", "bedrm", "master"]),
         ("dining_room", ["dining"]), ("living_room", ["living", "lounge"]),
         ("corridor_or_hallway", ["corridor", "hall", "entrance", "entry", "passage", "lobby"]),
         ("storage", ["store", "cupboard", "closet", "pantry"]), ("study_or_office", ["study", "office"]),
         ("laundry_or_utility", ["laundry", "utility"])]
AREA_RX = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m2|m²|sqm|sq\.?m)\b", re.I)
DIM_RX = re.compile(r"(\d+(?:\.\d+)?)\s*m?\s*[x×]\s*(\d+(?:\.\d+)?)\s*m", re.I)


def _fill_small_holes(poly, frac=0.015):
    """Holes smaller than `frac` of the room are bold glyphs / fixtures that survived the wall-thickness filter, not boundaries."""
    keep = [h for h in poly.interiors if Polygon(h).area > frac * poly.area]
    return Polygon(poly.exterior, keep)


def label_type(text):
    t = re.sub(r"[^a-z0-9.]", "", text.lower())
    for cls, kws in RULES:
        if any(k in t for k in kws):
            return cls
    return None


def printed_area(text):
    m = AREA_RX.search(text.replace("²", "2"))
    if m:
        return float(m.group(1))
    m = DIM_RX.search(text)
    return float(m.group(1)) * float(m.group(2)) if m else None


def load_embedded_image(page):
    doc = page.parent
    xref = page.get_images()[0][0]
    pix = pymupdf.Pixmap(doc, xref)
    if pix.colorspace is not None and pix.colorspace.n >= 3:
        pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, 0].copy()
    rect = page.get_image_rects(xref)[0]
    return a, rect


def thick_wall_mask(gray, close_k=3, open_frac=0.0075):
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (gray < t).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))   # joins dithered / hatched wall texture
    k = max(5, int(round(gray.shape[1] * open_frac)))
    walls = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return ink, walls, float(t), k


def footprint_mask(ink, close_frac=0.04):
    k = max(9, int(round(ink.shape[1] * close_frac)) | 1)
    c = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    inv = (c == 0).astype(np.uint8)
    ff = inv.copy()
    cv2.floodFill(ff, np.zeros((inv.shape[0] + 2, inv.shape[1] + 2), np.uint8), (0, 0), 2)
    return (ff != 2).astype(np.uint8)      # everything not reachable from the image border = inside the drawing outline


def detect_door_arcs_raster(ink, walls, min_len=25, r_range=(0.03, 0.14), tol=2.0, iters=250, seed=0, anchor_px=None):
    """Door swings = thin (non-wall) strokes that fit a circle over 60-110 deg. RANSAC per connected thin component."""
    rng = np.random.default_rng(seed)
    W = ink.shape[1]
    near_wall = cv2.dilate(walls, np.ones((2 * (anchor_px or max(6, int(0.02 * W))) + 1,) * 2, np.uint8))
    thin = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.dilate(walls, np.ones((5, 5), np.uint8))))
    n, lab, st, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    out = []
    for i in range(1, n):
        if st[i, 4] < min_len or st[i, 2] > 0.35 * W:
            continue
        ys, xs = np.nonzero(lab == i)
        P = np.c_[xs, ys].astype(float)
        if len(P) < min_len:
            continue
        best = None
        for _ in range(iters):
            a, b, c = P[rng.choice(len(P), 3, replace=False)]
            A = np.array([[b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]]) * 2
            B = np.array([b @ b - a @ a, c @ c - a @ a])
            if abs(np.linalg.det(A)) < 1e-6:
                continue
            ctr = np.linalg.solve(A, B)
            r = np.linalg.norm(a - ctr)
            if not (r_range[0] * W <= r <= r_range[1] * W):
                continue
            inl = np.abs(np.linalg.norm(P - ctr, axis=1) - r) < tol
            if best is None or inl.sum() > best[0]:
                best = (int(inl.sum()), ctr, r, inl)
        if best is None or best[0] < max(min_len, 0.5 * len(P) * 0.4):
            continue
        _, ctr, r, inl = best
        ang = np.degrees(np.arctan2(P[inl, 1] - ctr[1], P[inl, 0] - ctr[0]))
        ref = np.degrees(np.arctan2(np.sin(np.radians(ang)).mean(), np.cos(np.radians(ang)).mean()))
        rel = (ang - ref + 180) % 360 - 180
        sweep = rel.max() - rel.min()
        hx, hy = int(round(ctr[0])), int(round(ctr[1]))
        anchored = 0 <= hy < near_wall.shape[0] and 0 <= hx < near_wall.shape[1] and near_wall[hy, hx] > 0     # the hinge sits on / next to a wall
        if 65 <= sweep <= 110 and best[0] >= 0.9 * r and best[0] >= 0.45 * len(P) and anchored:
            out.append(dict(hinge_px=[round(float(ctr[0]), 1), round(float(ctr[1]), 1)], leaf_radius_px=round(float(r), 1), swing_sweep_deg=round(float(sweep), 1),
                            inlier_px=int(best[0]), source="raster:ransac_arc"))
    return out


def run_ocr_image(gray, cfg, device):
    if device == "cuda":
        import torch  # noqa: F401
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    from rapidocr_onnxruntime import RapidOCR
    from ..supporting_signals.ocr import _providers
    import unicodedata
    eng = RapidOCR(det_use_cuda=device == "cuda", cls_use_cuda=device == "cuda", rec_use_cuda=device == "cuda")
    out = []

    def iou(a, b):
        ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
        i = ix * iy
        return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)
    # several upscales: a label unreadable at one scale (e.g. 'WC' read as 'Iwd') is often readable at another
    for up in cfg.get("upscales", [3.0, 4.5]):
        big = cv2.resize(gray, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        res, _ = eng(cv2.cvtColor(big, cv2.COLOR_GRAY2BGR))
        for box, text, score in (res or []):
            if float(score) < cfg.get("min_score", 0.5) or not text.strip():
                continue
            b = np.array(box, float) / up
            bb = [b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()]
            txt = unicodedata.normalize("NFKC", text).strip()
            dup = next((o for o in out if iou(o["bbox"], bb) > 0.3), None)
            if dup is None:
                out.append(dict(text=txt, score=round(float(score), 3), bbox=bb, scale=up))
            elif (label_type(txt) and not label_type(dup["text"])) or (bool(label_type(txt)) == bool(label_type(dup["text"])) and float(score) > dup["score"]):
                dup.update(text=txt, score=round(float(score), 3), bbox=bb, scale=up)      # prefer a readable room label / higher confidence
    return out, _providers(eng)


def run_raster(pdf, cfg, classes, out_dir, log=print):
    t0 = time.time()
    out = Path(out_dir)
    (out / "debug").mkdir(parents=True, exist_ok=True)
    dev, dev_info = resolve_device(cfg.get("compute", {}).get("device", "auto"))
    log(f"compute device: {dev} {dev_info}")
    rc = cfg.get("raster", {})
    scale_r = rc.get("render_scale", 2.0)
    doc = pymupdf.open(pdf)
    page = doc[0]
    gray, rect = load_embedded_image(page)
    H, W = gray.shape
    # image px -> page pt (axis-aligned placement; flips/rotation are not handled and would be rejected)
    sx, sy = rect.width / W, rect.height / H
    to_pt = lambda p: np.array(p, float) * [sx, sy] + [rect.x0, rect.y0]
    log(f"[1/5] raster-only PDF: embedded image {W}x{H}px placed at {[round(v, 1) for v in rect]} pt on a {page.rect.width:.0f}x{page.rect.height:.0f} pt page")

    ink, walls, thr, k = thick_wall_mask(gray)
    foot = footprint_mask(ink, rc.get("footprint_close_frac", 0.04))
    log(f"[2/5] Otsu threshold {thr:.0f}, wall opening kernel {k}px, footprint {100 * foot.mean():.0f}% of image")

    log("[3/5] OCR (labels = room seeds)")
    texts, providers = run_ocr_image(gray, cfg.get("ocr", {}), dev)
    write_json(out / "debug" / "ocr.json", texts)
    log(f"      text boxes={len(texts)}  providers={sorted({p for v in providers.values() for p in v})}")

    # --- seeds -------------------------------------------------------------------------------------------------
    seeds = []
    for t in texts:
        cls = label_type(t["text"])
        if cls:
            cx, cy = (t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2
            seeds.append(dict(cls=cls, text=t["text"], xy=(cx, cy), bbox=t["bbox"]))
    # free space = footprint minus thick walls; labels sit on thin text pixels which are NOT barriers
    free = (foot > 0) & (walls == 0)
    dt = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 3)
    for i, s in enumerate(seeds, 1):
        x, y = int(s["xy"][0]), int(s["xy"][1])
        if 0 <= x < W and 0 <= y < H and not free[y, x]:         # snap into free space
            ys, xs = np.nonzero(free[max(0, y - 12):y + 13, max(0, x - 12):x + 13])
            if len(xs):
                j = np.argmin((xs - min(x, 12)) ** 2 + (ys - min(y, 12)) ** 2)
                x, y = max(0, x - 12) + int(xs[j]), max(0, y - 12) + int(ys[j])
        s["seed_px"] = (x, y)
    # geodesic nearest-seed partition: walls are impassable, narrow passages (door necks) are expensive, so cuts fall at doors
    # and, in open-plan areas without walls, midway between the labels
    alpha = rc.get("neck_penalty", 40.0)
    cost = np.full((H, W), np.inf)
    cost[free] = 1.0 + alpha / (dt[free] + 1.0)
    best = np.full((H, W), np.inf)
    ws = np.zeros((H, W), np.int32)
    for i, s_ in enumerate(seeds, 1):
        mcp = MCP_Geometric(cost, fully_connected=True)
        cum, _ = mcp.find_costs([(s_["seed_px"][1], s_["seed_px"][0])])
        upd = cum < best
        best[upd] = cum[upd]
        ws[upd] = i
    ws[~np.isfinite(best)] = 0
    log(f"[4/5] {len(seeds)} labelled seeds -> geodesic partition")

    # --- rooms -------------------------------------------------------------------------------------------------
    rooms, pt2 = [], None
    pxarea_to_pt2 = sx * sy
    # merge seeds that share a basin only if same class (e.g. a label split over two OCR boxes)
    for i, s in enumerate(seeds, 1):
        m = (ws == i)
        if m.sum() < 60:
            continue
        # open to drop thin tendrils that leak through door gaps, keep the largest piece containing the seed
        mo = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(mo, connectivity=4)
        if n < 2:
            continue
        sx0, sy0 = s["seed_px"]
        lid = lab[sy0, sx0] if lab[sy0, sx0] else 1 + int(np.argmax(st[1:, 4]))
        x, y, w, h, a = st[lid]
        poly = region_polygon(lab, lid, (x, y, w, h), (0, 0), 1.0, simplify_pt=1.0)
        if poly is None or poly.area < 120:
            continue
        poly = _fill_small_holes(poly)
        rooms.append(dict(seed=s, poly=poly, seed_index=i))
    # printed areas -> implied scale (validation signal, never used to alter polygons)
    # enclosed free space that no label basin reached (no door gap to any labelled room, or an unreadable label): report it, don't drop it
    left = ((free) & (ws == 0)).astype(np.uint8)
    left = cv2.morphologyEx(left, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    nL, labL, stL, _ = cv2.connectedComponentsWithStats(left, connectivity=4)
    min_a = rc.get("min_unlabelled_area_frac", 0.0008) * H * W
    for j in range(1, nL):
        x, y, w, h, a = stL[j]
        if a < min_a or x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        if cv2.distanceTransform((labL == j).astype(np.uint8), cv2.DIST_L2, 3).max() < 8:
            continue
        pg = region_polygon(labL, j, (x, y, w, h), (0, 0), 1.0, simplify_pt=1.0)
        if pg is not None and pg.area >= min_a:
            pg = _fill_small_holes(pg)
            rooms.append(dict(seed=dict(cls="unknown", text="", xy=(x + w / 2, y + h / 2)), poly=pg, seed_index=None, unlabelled=True))
    # boundaries that exist only because of label proximity (no wall between the two basins) = open-plan partition, not drawn geometry
    near_wall = cv2.dilate(walls, np.ones((9, 9), np.uint8)) > 0
    for r in rooms:
        i = r["seed_index"]
        if i is None:
            r["open_boundary_frac"] = 0.0
            continue
        m = (ws == i)
        edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        nb = cv2.dilate(((ws > 0) & (ws != i)).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        shared = edge & nb
        total = max(int(edge.sum()), 1)
        r["open_boundary_frac"] = round(float((shared & ~near_wall).sum()) / total, 2)
    printed = []
    for t in texts:
        a = printed_area(t["text"])
        if a:
            printed.append((a, ((t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2)))
    for r in rooms:
        r["printed_m2"] = next((a for a, c in printed if r["poly"].contains(Point(c))), None)
    ratios = [r["printed_m2"] / r["poly"].area for r in rooms if r["printed_m2"]]
    m2_per_px2 = float(np.median(ratios)) if ratios else None

    base = cv2.imdecode(np.frombuffer(page.get_pixmap(matrix=pymupdf.Matrix(scale_r, scale_r)).tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    PH, PW = base.shape[:2]
    recs = []
    for idx, r in enumerate(sorted(rooms, key=lambda r: (round(r["seed"]["xy"][1] / 80), r["seed"]["xy"][0])), 1):
        poly = r["poly"]
        ext = np.array(poly.exterior.coords)[:-1]
        pt = np.array([to_pt(p) for p in ext])
        holes_px = [np.array(h.coords)[:-1] for h in poly.interiors]
        area_img = float(poly.area)
        rec = dict(id=f"room-{idx:04d}", type=r["seed"]["cls"], polygon=(pt * scale_r).round(1).tolist(), polygon_pt=pt.round(2).tolist(),
                   holes=[(np.array([to_pt(p) for p in h]) * scale_r).round(1).tolist() for h in holes_px],
                   polygon_image_px=ext.round(1).tolist(), area_pixels=round(area_img * pxarea_to_pt2 * scale_r ** 2, 1), area_image_px=round(area_img, 1),
                   confidence=None, confidence_source="none: rule-based, no calibrated probability",
                   heuristic_score=(None if r.get("unlabelled") else 0.8),
                   evidence=(["enclosed_free_space_no_label_reached"] if r.get("unlabelled") else [f"ocr_label:{r['seed']['text']}"]),
                   flags=(["unlabelled_enclosed_space"] if r.get("unlabelled") else []), text_inside=([] if r.get("unlabelled") else [r["seed"]["text"]]), source="raster+ocr",
                   printed_area_m2=r["printed_m2"], approx_area_m2=(round(area_img * m2_per_px2, 1) if m2_per_px2 else None),
                   review_required=bool(r.get("unlabelled")))
        if r["open_boundary_frac"] >= 0.15:
            rec["flags"].append(f"open_plan_boundary_by_label_proximity:{int(100 * r['open_boundary_frac'])}pct_of_perimeter")
            rec["review_required"] = True
        if r["printed_m2"] and m2_per_px2:
            dev_pct = 100 * (area_img * m2_per_px2 - r["printed_m2"]) / r["printed_m2"]
            rec["area_vs_printed_pct"] = round(dev_pct, 1)
            if abs(dev_pct) > 25:
                rec["flags"].append(f"area_differs_from_printed_label_by_{dev_pct:.0f}pct")
                rec["review_required"] = True
        recs.append(rec)

    ink2, _, _ = ink, None, None
    doors = detect_door_arcs_raster(ink, walls)
    elements = []
    for d in doors:
        elements.append(dict(type="door", hinge_pt=to_pt(d["hinge_px"]).round(2).tolist(), leaf_radius_pt=round(d["leaf_radius_px"] * sx, 2),
                             swing_sweep_deg=d["swing_sweep_deg"], hinge_image_px=d["hinge_px"], source=d["source"]))
    cnts, hier = cv2.findContours(walls, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1 or len(c) < 3 or cv2.contourArea(c) < 80:
            continue
        pg = Polygon(c[:, 0, :].astype(float))
        pg = pg if pg.is_valid else pg.buffer(0)
        for part in ([pg] if pg.geom_type == "Polygon" else list(pg.geoms)):
            if part.area >= 80:
                e = np.array(part.simplify(1.0).exterior.coords)[:-1]
                elements.append(dict(type="wall", polygon_pt=np.array([to_pt(p) for p in e]).round(2).tolist(), polygon_image_px=e.round(1).tolist(), source="raster:thick_stroke_mask"))

    used = {}
    for r in recs:
        used[r["type"]] = used.get(r["type"], 0) + 1
    ov = draw_overlay(base, recs, classes, elements, [0.0, 0.0, PW / scale_r, PH / scale_r], scale_r)
    ov, legend = add_legend_inset(ov, classes, used, [("red ring = review required", (0, 0, 255)), ("red dot = door hinge", (0, 0, 220))])
    assert ov.shape == base.shape
    cv2.imwrite(str(out / "overlay.png"), ov)
    mk = np.full(base.shape, 255, np.uint8)
    for r in recs:
        cv2.fillPoly(mk, [np.array(r["polygon"]).astype(np.int32)], tuple(int(c) for c in classes[r["type"]]["color"][::-1]))
        for h in r["holes"]:
            cv2.fillPoly(mk, [np.array(h).astype(np.int32)], (255, 255, 255))
    cv2.imwrite(str(out / "room_masks.png"), mk)
    cv2.imwrite(str(out / "debug" / "render_reference.png"), base)
    cv2.imwrite(str(out / "debug" / "wall_mask_native.png"), 255 - walls * 255)
    dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ws_col = np.zeros_like(dbg)
    rng = np.random.default_rng(3)
    for i in range(1, len(seeds) + 1):
        ws_col[ws == i] = rng.integers(60, 255, 3)
    cv2.imwrite(str(out / "debug" / "watershed_native.png"), cv2.addWeighted(dbg, 0.5, ws_col, 0.5, 0))

    image = dict(width=PW, height=PH, coordinate_origin="top_left", units="pixels", output_canvas="full_page", legend=legend,
                 source_image=dict(width=W, height=H, placed_rect_pt=[round(v, 2) for v in rect], note="polygon_image_px are in this embedded image's pixels"),
                 transform=dict(type="pdf_page_render", scale_px_per_pt=scale_r, page_rotation_applied_deg=page.rotation, page_size_pt=[page.rect.width, page.rect.height],
                                pt_to_px="px = pt * scale_px_per_pt", image_px_to_pt="pt = rect.xy + image_px * (rect.size / image.size)"),
                 note="Raster-only drawing. approx_area_m2 is calibrated from the drawing's own printed room areas (median ratio) and is only approximate; null if none were found.")
    write_json(out / "rooms.json", dict(image=image, rooms=recs))
    write_json(out / "elements.json", dict(image=image, elements=elements))
    labelled = sum(1 for t in texts if label_type(t["text"]))
    cv_pct = None
    if len(ratios) > 2:
        a = np.array(ratios)
        cv_pct = round(float(100 * a.std() / a.mean()), 1)
    stats = dict(mode="raster", device=dev, device_info=dev_info, ocr_providers=providers, runtime_s=round(time.time() - t0, 1), n_rooms=len(recs), by_type=used,
                 n_elements={"door": len(doors), "wall": sum(e["type"] == "wall" for e in elements)}, ocr_boxes=len(texts), room_label_boxes=labelled,
                 rooms_with_printed_area=len(ratios), implied_m2_per_px2=m2_per_px2, printed_area_ratio_cv_pct=cv_pct,
                 note="classical CV + OCR; no trained room model, so no per-room probabilities")
    write_json(out / "run_stats.json", stats)
    log(f"[5/5] done in {stats['runtime_s']}s: {stats}")
    return recs, elements, stats
