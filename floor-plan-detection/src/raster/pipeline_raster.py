"""Raster-only floor plans (an image embedded in a PDF): classical CV + OCR, no trained room model.

v4 pipeline: OCR (labels = seeds, all text = suppressed from the wall mask) -> exact wall mask -> wall-gap closure (door thresholds /
openings along the wall axis) -> plan footprint from the wall cluster -> enclosed free-space components = rooms (open-plan components with
several labels are split by a geodesic partition) -> pixel-exact polygons with holes, axis snapping, shape descriptors -> door arcs and
circles. Polygons are written in three frames: embedded-image px, PDF points, full-page render px (see PageTransform)."""
import re
import time
import unicodedata
from pathlib import Path
import numpy as np
import cv2
import pymupdf
from shapely.geometry import Polygon, Point

from ..output.json_writer import write_json
from ..output.visualization import draw_overlay, add_legend_inset
from ..postprocessing.shapes import describe, snap_axis_aligned
from ..gpu import resolve_device, peak_ram_mb, GpuSampler
from .segmentation import PageTransform, segment_rooms

# ordered: first match wins ("Kitchen veranda" must be a balcony, "Covered Veranda" too)
RULES = [("balcony", ["veranda", "porch", "balcony", "terrace"]), ("toilet_or_wc", ["wc", "w.c", "toilet"]),
         ("bathroom", ["bath", "shower", "ensuite"]), ("kitchen", ["kitchen"]), ("bedroom", ["bedroom", "bedrm", "master"]),
         ("dining_room", ["dining"]), ("living_room", ["living", "lounge"]),
         ("corridor_or_hallway", ["corridor", "hall", "entrance", "entry", "passage", "lobby"]),
         ("storage", ["store", "cupboard", "closet", "pantry"]), ("study_or_office", ["study", "office"]),
         ("laundry_or_utility", ["laundry", "utility"])]
AREA_RX = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m2|m²|sqm|sq\.?m)\b", re.I)
DIM_RX = re.compile(r"(\d+(?:\.\d+)?)\s*m?\s*[x×]\s*(\d+(?:\.\d+)?)\s*m", re.I)


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


# ----------------------------------------------------------------------------------------------------- doors / circles
def detect_door_arcs_raster(ink, walls, min_len=25, r_range=(0.03, 0.14), tol=2.0, iters=250, seed=0, anchor_px=None):
    """Door swings = thin (non-wall) strokes that fit a circle over 65-110 deg with the hinge on / next to a wall (RANSAC)."""
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
        if best is None:
            continue
        _, ctr, r, inl = best
        ang = np.degrees(np.arctan2(P[inl, 1] - ctr[1], P[inl, 0] - ctr[0]))
        ref = np.degrees(np.arctan2(np.sin(np.radians(ang)).mean(), np.cos(np.radians(ang)).mean()))
        rel = (ang - ref + 180) % 360 - 180
        sweep = rel.max() - rel.min()
        hx, hy = int(round(ctr[0])), int(round(ctr[1]))
        anchored = 0 <= hy < near_wall.shape[0] and 0 <= hx < near_wall.shape[1] and near_wall[hy, hx] > 0
        if 65 <= sweep <= 110 and best[0] >= 0.9 * r and best[0] >= 0.45 * len(P) and anchored:
            out.append(dict(hinge_px=[round(float(ctr[0]), 1), round(float(ctr[1]), 1)], leaf_radius_px=round(float(r), 1),
                            swing_sweep_deg=round(float(sweep), 1), inlier_px=int(best[0]), source="raster:ransac_arc"))
    return out


def detect_circles_raster(gray, ink_nt, walls, text_boxes=(), r_range=(0.007, 0.03), min_cover=0.88, max_resid=1.3):
    """Closed circular outlines (columns, round fixtures): Hough proposals, verified by ring ink coverage and radial residual.
    Partial arcs (door swings) fail the coverage test; wall pixels and OCR text are excluded."""
    H, W = gray.shape
    g = cv2.GaussianBlur(gray, (5, 5), 1.2)
    rmin, rmax = max(3, int(r_range[0] * W)), int(r_range[1] * W)
    cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=2 * rmin, param1=120, param2=16, minRadius=rmin, maxRadius=rmax)
    out = []
    if cs is None:
        return out
    ang = np.deg2rad(np.arange(0, 360, 4))
    dist_ink = cv2.distanceTransform((ink_nt == 0).astype(np.uint8), cv2.DIST_L2, 3)
    for cx, cy, r in cs[0]:
        px, py = cx + r * np.cos(ang), cy + r * np.sin(ang)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if ok.mean() < 0.95:
            continue
        xi, yi = np.clip(px.astype(int), 0, W - 1), np.clip(py.astype(int), 0, H - 1)
        d = dist_ink[yi, xi]
        cover = float((d <= 1.5).mean())
        if cover < min_cover or float(d[d <= 3].mean() if (d <= 3).any() else 9) > max_resid:
            continue
        if walls[int(cy), int(cx)] or any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in text_boxes):
            continue
        out.append(dict(center_px=[round(float(cx), 1), round(float(cy), 1)], radius_px=round(float(r), 1), ring_coverage=round(cover, 3), source="raster:hough+ring_fit"))
    keep = []                                                                             # merge near-duplicates
    for c in sorted(out, key=lambda c: -c["ring_coverage"]):
        if all(np.hypot(c["center_px"][0] - k["center_px"][0], c["center_px"][1] - k["center_px"][1]) > 0.6 * min(c["radius_px"], k["radius_px"]) for k in keep):
            keep.append(c)
    return keep


# ------------------------------------------------------------------------------------------------------------- OCR
def run_ocr_image(gray, cfg, device):
    if device == "cuda":
        import torch  # noqa: F401
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    from rapidocr_onnxruntime import RapidOCR
    from ..supporting_signals.ocr import _providers
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
                dup.update(text=txt, score=round(float(score), 3), bbox=bb, scale=up)
    return out, _providers(eng)


# --------------------------------------------------------------------------------------------------------- pipeline
def run_raster(pdf, cfg, classes, out_dir, log=print):
    t0 = time.time()
    T = {}
    _t = [time.time()]

    def tick(name):
        now = time.time()
        T[name] = round(now - _t[0], 2)
        _t[0] = now
    out = Path(out_dir)
    (out / "debug").mkdir(parents=True, exist_ok=True)
    dev, dev_info = resolve_device(cfg.get("compute", {}).get("device", "auto"))
    log(f"compute device: {dev} {dev_info}")
    sampler = GpuSampler(cfg.get("compute", {}).get("sample_interval_s", 0.1)) if (dev == "cuda" and cfg.get("compute", {}).get("sample_gpu", True)) else None
    if sampler:
        sampler.__enter__()
    rc = cfg.get("raster", {})
    scale_r = rc.get("render_scale", 2.0)
    doc = pymupdf.open(pdf)
    page = doc[0]
    gray, rect = load_embedded_image(page)
    H, W = gray.shape
    tf = PageTransform(tuple(rect), W, H, scale_r)
    log(f"[1/5] raster-only PDF: embedded image {W}x{H}px placed at {[round(v, 1) for v in rect]} pt on a {page.rect.width:.0f}x{page.rect.height:.0f} pt page")
    tick("load")

    log("[2/5] OCR (labels = room seeds; ALL text is removed from the wall mask)")
    texts, providers = run_ocr_image(gray, cfg.get("ocr", {}), dev)
    write_json(out / "debug" / "ocr.json", texts)
    tick("ocr")
    log(f"      text boxes={len(texts)}  providers={sorted({p for v in providers.values() for p in v})}")
    boxes = [t["bbox"] for t in texts]
    seeds = []
    for t in texts:
        cls = label_type(t["text"])
        if cls:
            seeds.append(dict(cls=cls, text=t["text"], xy=((t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2)))

    log("[3/5] walls -> gap closure -> footprint -> rooms")
    seg = segment_rooms(gray, seeds, boxes, rc)
    info = seg["info"]
    log(f"      Otsu {info['otsu']:.0f}, wall kernel {info['wall_kernel_px']}px, text blobs removed {info['text_blobs_removed']}, "
        f"gap closures {info['n_closures']}, footprint {100 * info['footprint_frac']:.0f}% of image, {len(seg['rooms'])} rooms from {len(seeds)} labels")
    tick("segmentation")

    # printed areas -> implied scale (validation signal only; never used to alter polygons)
    printed = [(printed_area(t["text"]), ((t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2)) for t in texts if printed_area(t["text"])]
    rooms = []
    for r in seg["rooms"]:
        poly = snap_axis_aligned(r["poly"])
        sid = r["seed_ids"][0] if r["seed_ids"] else None
        s = seeds[sid] if sid is not None else None
        rooms.append(dict(poly=poly, raw_poly=r["poly"], seed=s, kind=r["kind"], open_boundary_frac=r["open_boundary_frac"],
                          printed_m2=next((a for a, c in printed if poly.contains(Point(c))), None)))
    ratios = [r["printed_m2"] / r["poly"].area for r in rooms if r["printed_m2"]]
    m2_per_px2 = float(np.median(ratios)) if ratios else None

    base = cv2.imdecode(np.frombuffer(page.get_pixmap(matrix=pymupdf.Matrix(scale_r, scale_r)).tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    PH, PW = base.shape[:2]
    recs = []
    for idx, r in enumerate(sorted(rooms, key=lambda r: (round(r["poly"].centroid.y / 80), r["poly"].centroid.x)), 1):
        poly = r["poly"]
        ext = np.array(poly.exterior.coords)[:-1]
        holes = [np.array(h.coords)[:-1] for h in poly.interiors]
        unl = r["seed"] is None
        a_img = float(poly.area)
        rec = dict(id=f"room-{idx:04d}", type=("unknown" if unl else r["seed"]["cls"]),
                   polygon=tf.img_to_page_px(ext).round(1).tolist(), polygon_pt=tf.img_to_pt(ext).round(2).tolist(),
                   holes=[tf.img_to_page_px(h).round(1).tolist() for h in holes], polygon_image_px=ext.round(2).tolist(),
                   holes_image_px=[h.round(2).tolist() for h in holes],
                   area_pixels=round(a_img * tf.sx * tf.sy * scale_r ** 2, 1), area_image_px=round(a_img, 1),
                   shape=describe(poly), segmentation=r["kind"],
                   confidence=None, confidence_source="none: rule-based, no calibrated probability", heuristic_score=(None if unl else 0.8),
                   evidence=(["enclosed_free_space_no_label_reached"] if unl else [f"ocr_label:{r['seed']['text']}"]),
                   flags=(["unlabelled_enclosed_space"] if unl else []), text_inside=([] if unl else [r["seed"]["text"]]), source="raster+ocr",
                   printed_area_m2=r["printed_m2"], approx_area_m2=(round(a_img * m2_per_px2, 1) if m2_per_px2 else None), review_required=unl)
        if r["kind"] == "bounded_by_thin_outline":
            rec["flags"].append("bounded_by_thin_dashed_outline_not_a_wall")
            rec["review_required"] = True
        if r["kind"] == "split":
            rec["flags"].append(f"open_plan_split_by_label_proximity:{int(100 * r['open_boundary_frac'])}pct_of_perimeter")
            rec["review_required"] = True
        if r["printed_m2"] and m2_per_px2:
            dev_pct = 100 * (a_img * m2_per_px2 - r["printed_m2"]) / r["printed_m2"]
            rec["area_vs_printed_pct"] = round(dev_pct, 1)
            if abs(dev_pct) > 25:
                rec["flags"].append(f"area_differs_from_printed_label_by_{dev_pct:.0f}pct")
                rec["review_required"] = True
        recs.append(rec)
    tick("polygons")

    elements = []
    doors = detect_door_arcs_raster(seg["ink"], seg["walls"])
    for d in doors:
        elements.append(dict(type="door", hinge_pt=tf.img_to_pt(d["hinge_px"]).round(2).tolist(), leaf_radius_pt=round(d["leaf_radius_px"] * tf.sx, 2),
                             swing_sweep_deg=d["swing_sweep_deg"], hinge_image_px=d["hinge_px"], source=d["source"]))
    circles = detect_circles_raster(gray, seg["ink"], seg["walls"], boxes)
    for c in circles:
        elements.append(dict(type="circle", center_pt=tf.img_to_pt(c["center_px"]).round(2).tolist(), radius_pt=round(c["radius_px"] * tf.sx, 2),
                             center_image_px=c["center_px"], radius_image_px=c["radius_px"], ring_coverage=c["ring_coverage"], source=c["source"]))
    for cl in seg["closures"]:
        elements.append(dict(type="opening_line", p0_pt=tf.img_to_pt(cl["p0"]).round(2).tolist(), p1_pt=tf.img_to_pt(cl["p1"]).round(2).tolist(),
                             p0_image_px=cl["p0"], p1_image_px=cl["p1"], gap_px=cl["gap_px"], kind=cl["kind"], source="raster:wall_end_extension"))
    cnts, hier = cv2.findContours(seg["walls"], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1 or len(c) < 3 or cv2.contourArea(c) < 80:
            continue
        pg = Polygon(c[:, 0, :].astype(float))
        pg = pg if pg.is_valid else pg.buffer(0)
        for part in ([pg] if pg.geom_type == "Polygon" else list(pg.geoms)):
            if part.area >= 80:
                e = np.array(part.simplify(0.8).exterior.coords)[:-1]
                elements.append(dict(type="wall", polygon_pt=tf.img_to_pt(e).round(2).tolist(), polygon_image_px=e.round(1).tolist(), source="raster:thick_stroke_mask"))
    tick("elements")

    used = {}
    for r in recs:
        used[r["type"]] = used.get(r["type"], 0) + 1
    ov = draw_overlay(base, recs, classes, elements, [0.0, 0.0, PW / scale_r, PH / scale_r], scale_r)
    ov, legend = add_legend_inset(ov, classes, used, [("red ring = review required", (0, 0, 255)), ("red dot = door hinge", (0, 0, 220)), ("magenta line = opening / threshold", (200, 0, 200)), ("black circle = circular shape", (20, 20, 20))])
    assert ov.shape == base.shape
    cv2.imwrite(str(out / "overlay.png"), ov)
    mk = np.full(base.shape, 255, np.uint8)
    for r in recs:
        colr = tuple(int(c) for c in classes[r["type"]]["color"][::-1])
        cv2.fillPoly(mk, [np.round(np.array(r["polygon"])).astype(np.int32)], colr)
        for h in r["holes"]:
            cv2.fillPoly(mk, [np.round(np.array(h)).astype(np.int32)], (255, 255, 255))
    cv2.imwrite(str(out / "room_masks.png"), mk)
    cv2.imwrite(str(out / "debug" / "render_reference.png"), base)
    cv2.imwrite(str(out / "debug" / "wall_mask_native.png"), 255 - seg["walls"] * 255)
    cv2.imwrite(str(out / "debug" / "wall_mask_closed_native.png"), 255 - seg["closed"] * 255)
    cv2.imwrite(str(out / "debug" / "footprint_native.png"), 255 - seg["footprint"] * 255)
    tick("write_images")

    image = dict(width=PW, height=PH, coordinate_origin="top_left", units="pixels", output_canvas="full_page", legend=legend,
                 source_image=dict(width=W, height=H, placed_rect_pt=[round(v, 2) for v in rect], note="*_image_px fields are in this embedded image's pixels"),
                 transform=dict(type="pdf_page_render", scale_px_per_pt=scale_r, page_rotation_applied_deg=page.rotation, page_size_pt=[page.rect.width, page.rect.height],
                                pt_to_px="px = pt * scale_px_per_pt", image_px_to_pt="pt = rect.xy + image_px * (rect.size / image.size)"),
                 note="Raster-only drawing. approx_area_m2 is calibrated from the drawing's own printed room areas (median ratio) and is only approximate; null if none were found.")
    write_json(out / "rooms.json", dict(image=image, rooms=recs))
    write_json(out / "elements.json", dict(image=image, elements=elements))
    labelled = sum(1 for t in texts if label_type(t["text"]))
    cv_pct = round(float(100 * np.std(ratios) / np.mean(ratios)), 1) if len(ratios) > 2 else None
    if sampler:
        sampler._stop.set()
    stats = dict(mode="raster", version="v4", device=dev, device_info=dev_info, ocr_providers=providers, gpu_measured=(sampler.summary() if sampler else None), stage_times_s=T, runtime_s=round(time.time() - t0, 1),
                 n_rooms=len(recs), by_type=used,
                 n_elements={k: sum(e["type"] == k for e in elements) for k in ("door", "circle", "opening_line", "wall")},
                 ocr_boxes=len(texts), room_label_boxes=labelled, segmentation=info, rooms_with_printed_area=len(ratios), implied_m2_per_px2=m2_per_px2,
                 printed_area_ratio_cv_pct=cv_pct, peak_ram_mb=peak_ram_mb(), note="classical CV + OCR; no trained room model, so no per-room probabilities")
    write_json(out / "run_stats.json", stats)
    log(f"[5/5] done in {stats['runtime_s']}s: {stats}")
    return recs, elements, stats
