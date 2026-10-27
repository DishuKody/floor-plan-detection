"""Vector-first floor-plan pipeline for CAD-exported PDFs (walls/doors/fixtures recoverable from vector paths)."""
import time
from pathlib import Path
import numpy as np
import cv2
import pymupdf
from shapely.geometry import box, Point, Polygon
from shapely.ops import unary_union

from .vector.extract_pdf_paths import extract_paths
from .vector.doors import detect_door_arcs
from .vector.walls import find_grid_paths, rasterize_mask, extend_dangling_walls, rasterize_thick
from .vector.candidate_regions import close_doors, label_regions
from .vector.symbols import find_tread_series
from .vector.calibrate import structural_stroke, plan_footprint
from .vector.fixtures_template import line_layer, find_symbol
from .vector.elements import door_elements, column_elements, window_elements, window_elements_template, circle_fixtures, circle_elements
from .postprocessing.polygons import region_polygon, polygon_to_record
from .postprocessing.shapes import describe
from .postprocessing.classify import classify_room, match_keywords, UNIT_ID
from .postprocessing.instances import split_region
from .postprocessing.structural import structural_rooms
from .output.visualization import draw_overlay, add_legend, add_legend_inset
from .output.json_writer import write_json
from .gpu import resolve_device, GpuSampler


def _peak_ram_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().peak_wset / 2 ** 20)
    except Exception:
        return None


def run_vector(pdf, cfg, classes, out_dir, log=print):
    # raster-only PDF (an embedded picture, no vector geometry): classical raster pipeline instead of the vector one
    _pg = pymupdf.open(pdf)[0]
    if len(_pg.get_drawings()) < 500 and _pg.get_images():
        from .raster.pipeline_raster import run_raster
        log(f"{Path(pdf).name}: no vector geometry ({len(_pg.get_drawings())} paths) and an embedded image -> raster mode")
        return run_raster(pdf, cfg, classes, out_dir, log)
    dev, dev_info = resolve_device(cfg.get("compute", {}).get("device", "auto"))
    log(f"compute device: {dev} {dev_info}")
    sampler = GpuSampler(interval=cfg.get("compute", {}).get("sample_interval_s", 0.25)) if (dev == "cuda" and cfg.get("compute", {}).get("sample_gpu", True)) else None
    if sampler:
        sampler.__enter__()
    try:
        return _run_vector(pdf, cfg, classes, out_dir, log, dev, dev_info, sampler)
    finally:
        if sampler:
            sampler._stop.set()


def _run_vector(pdf, cfg, classes, out_dir, log, dev, dev_info, sampler):
    t0 = time.time()
    T = {}
    _t = [time.time()]

    def tick(name):
        now = time.time()
        T[name] = round(now - _t[0], 2)
        _t[0] = now

    out = Path(out_dir)
    (out / "debug").mkdir(parents=True, exist_ok=True)
    V = cfg["vector"]
    scale_r = cfg["input"]["render_scale"]
    S = V["raster_px_per_pt"]
    doc = pymupdf.open(pdf)
    page = doc[0]
    ratio = cfg["drawing"]["scale_ratio"]
    # pt^2 per real m^2 at the printed drawing scale (1 pt = 25.4/72 mm on paper)
    pt2_per_m2 = (1000.0 / (25.4 / 72 * ratio)) ** 2
    # every length below that was tuned in pt on a 1:100 sheet is scaled to this sheet's printed scale (f = 1 at 1:100)
    f = 100.0 / ratio

    log("[1/6] extracting vector paths")
    P = extract_paths(page)
    if len(P) < 500:
        raise RuntimeError(
            f"{Path(pdf).name}: only {len(P)} vector paths ({len(page.get_images())} embedded raster image(s)). This is a raster-only "
            "drawing; the vector pipeline cannot be used. A raster model (e.g. Raster2Seq / segmentation) is required - see analysis_report.md.")
    # sheet calibration: explicit config values win, otherwise they are derived from the drawing itself (src/vector/calibrate.py)
    auto_w, dom_w, _ = structural_stroke(P)
    min_wall = V.get("min_wall_stroke") or auto_w
    cal = dict(printed_scale=f"1:{ratio}", length_factor=round(f, 3), dominant_line_weight_pt=dom_w, min_wall_stroke_pt=min_wall,
               min_wall_stroke_source="config" if V.get("min_wall_stroke") else "auto: 1.5 x dominant line weight")
    if V.get("footprint"):
        foot = unary_union([box(*b) for b in V["footprint"]])
        cal["footprint_source"] = "config"
    else:
        foot, finfo = plan_footprint(P, page.rect, V.get("footprint_stroke") or round(3.3 * dom_w, 3), ratio)
        if foot is None:
            raise RuntimeError(f"auto footprint failed: {finfo}")
        cal.update(footprint_source="auto", footprint_info=finfo)
    roi = tuple(V["roi"]) if V.get("roi") else tuple(float(v) for v in (
        max(foot.bounds[0] - 30 * f, 0), max(foot.bounds[1] - 30 * f, 0), min(foot.bounds[2] + 30 * f, page.rect.width), min(foot.bounds[3] + 30 * f, page.rect.height)))
    cal["roi_pt"] = [round(v, 1) for v in roi]
    grid = find_grid_paths(P)
    if V.get("skip_dashed", not V.get("roi")):
        grid |= {i for i, q in enumerate(P) if q.dashed}        # dash-dot grid / axis / boundary lines are not walls
    da = V["door_arc"]
    arcs = detect_door_arcs(P, r_range=(da["r_min"] * f, da["r_max"] * f), sweep_range=(da["sweep_min"], da["sweep_max"]))
    log(f"      paths={len(P)} grid_dash_paths_removed={len(grid)} door_arcs={len(arcs)} | calibration {cal}")
    tick("extract_paths_and_arcs")

    log("[2/6] wall mask, wall-end extension, door-opening closure")
    skip = grid | {j for a in arcs for j in a.path_indices}
    if V.get("remove_tread_series", False):
        treads = find_tread_series(P, skip)
        skip |= treads
        log(f"      stair-tread/hatch series removed from wall mask: {len(treads)} segments")
    mask = rasterize_mask(P, skip, roi, S, min_w=min_wall)
    ew = V["extend_walls"]
    ext = extend_dangling_walls(P, skip, mask, roi, S, min_len=ew["min_len"] * f, max_ext=ew["max_ext"] * f)
    chords_full = close_doors(mask, arcs, roi, S)
    arcs = [c[3] for c in chords_full]            # arcs accepted as doors (thin arcs need wall evidence)
    chords = [c[:3] for c in chords_full]
    log(f"      doors accepted={len(arcs)}")
    closed, lab, regs = label_regions(mask, S, V["wall_channel_close_pt"] * f, V["min_area_pt2"] * f * f, V["min_inscribed_radius_pt"] * f)
    gc = V.get("gap_closing", {})
    gap_lines = []
    if gc.get("enabled", False):
        # direction-agnostic wall-end extension (diagonal / curved plans): every skeleton end runs on along its own direction to the next wall
        from .raster.segmentation import close_wall_gaps
        sk_dt = cv2.distanceTransform(closed, cv2.DIST_L2, 3)
        k_px = max(3, int(round(2 * float(np.percentile(sk_dt[sk_dt > 0], 75)))))
        max_gap_px = gc.get("max_gap_m", 3.0) * 28.346 * f * S
        closed, gap_lines = close_wall_gaps(closed, k_px, max_gap_px / 0.11, door_frac=0.11, pair_frac=0.11, rectilinear_only=False)
        closed, lab, regs = label_regions(closed, S, 0.0, V["min_area_pt2"] * f * f, V["min_inscribed_radius_pt"] * f)
        log(f"      direction-agnostic gap closing: {len(gap_lines)} wall-end extensions (wall thickness ~{k_px / S:.1f} pt, max gap {gc.get('max_gap_m', 3.0)} m) -> {len(regs)} regions")
    log(f"      wall extensions={len(ext)} candidate regions={len(regs)}")
    tick("wall_mask_extension_doors_labeling")

    log("[3/6] region polygons")
    rooms_poly = []
    for r in regs:
        if r["border"]:
            continue
        poly = region_polygon(lab, r["label"], r["bbox"], roi, S)
        if poly is None or not foot.contains(poly.representative_point()):
            continue
        rooms_poly.append((poly, r))
    log(f"      rooms inside building footprint={len(rooms_poly)}")
    tick("region_polygons")

    texts = []
    if cfg["ocr"]["enabled"]:
        log("[4/6] OCR (local RapidOCR/ONNX)")
        from .supporting_signals.ocr import run_ocr
        ocr_info = {}
        texts = run_ocr(page, roi, cfg["ocr"], dev, ocr_info)
        log(f"      OCR providers: {ocr_info.get('ocr_providers')}")
        write_json(out / "debug" / "ocr.json", texts)
        log(f"      text boxes={len(texts)}")
    tick("ocr")
    circles = circle_fixtures(P, f=f, w_range=(0.4 * dom_w, 1.25 * dom_w))

    n_split = 0
    if cfg.get("instances", {}).get("split_by_labels", False) and texts:
        unit_pos = {}
        for t in texts:
            m = UNIT_ID.search(t["text"])
            if m:
                unit_pos.setdefault(m.group(0).upper(), []).append(((t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2))
        unit_pos = {k: np.mean(v, axis=0) for k, v in unit_pos.items()}
        new = []
        for poly, r in rooms_poly:
            seeds = [p for p in unit_pos.values() if poly.contains(Point(p))]
            ok = len(seeds) >= 2 and poly.area / pt2_per_m2 <= 12 * len(seeds) + 10                 and min(np.linalg.norm(a - b) for i, a in enumerate(seeds) for b in seeds[i + 1:]) >= 25
            parts = split_region(lab, r["label"], r["bbox"], seeds, roi, S) if ok else None
            if parts and all(p is not None and not p.is_empty for p in parts):
                n_split += 1
                new += [(p, dict(r, split_from=r["label"], n_split=len(parts))) for p in parts]
            else:
                new.append((poly, r))
        rooms_poly = new
        log(f"      label-guided watershed split {n_split} merged regions -> {len(rooms_poly)} rooms")

    osp = cfg.get("instances", {}).get("open_plan_split", {})
    if osp.get("enabled") and texts:
        # a large free-space region that holds several room labels is an open-plan merge (rooms open to a concourse / corridor without a drawn
        # separator): split it with a marker-controlled watershed from the labels, plus one seed for the open space itself. Flagged for review.
        from shapely.ops import polylabel
        room_cls = {n: sp for n, sp in classes.items() if n not in ("corridor_or_hallway", "communal_or_shared_space", "unknown", "other", "reception_or_entrance")}
        near = osp.get("cluster_m", 4.0) * 28.346 * f
        new, n_open = [], 0
        for poly, r in rooms_poly:
            if poly.area / pt2_per_m2 < osp.get("min_area_m2", 150):
                new.append((poly, r)); continue
            pts = []
            for t in texts:
                c = np.array([(t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2])
                if poly.contains(Point(c)) and match_keywords(t["text"], room_cls)[0]:
                    for q in pts:                    # words of one label ("Design 01" / "Tech" / "Engineering") share a seed
                        if np.linalg.norm(q[0] / q[1] - c) < near:
                            q[0] += c; q[1] += 1; break
                    else:
                        pts.append([c.copy(), 1])
            seeds = [q[0] / q[1] for q in pts]
            if len(seeds) < osp.get("min_seeds", 2):
                new.append((poly, r)); continue
            parts = split_region(lab, r["label"], r["bbox"], seeds, roi, S)
            if parts and all(g is not None and not g.is_empty for g in parts):
                n_open += 1
                # a room does not reach further than `reach_m` from its label: what is left of the region is the open space (concourse / circulation)
                reach = osp.get("reach_m", 9.0) * 28.346 * f
                mine = []
                for g, sd in zip(parts, seeds):
                    g = g.intersection(Point(sd).buffer(reach))
                    g = g if g.geom_type == "Polygon" else max(g.geoms, key=lambda q: q.area)
                    mine.append(g)
                rest = poly.difference(unary_union(mine).buffer(0.3))
                new += [(g, dict(r, split_from=r["label"], n_split=len(parts), open_plan=True)) for g in mine]
                new += [(g, dict(r, split_from=r["label"], n_split=len(parts), open_plan=True, open_core=True))
                        for g in getattr(rest, "geoms", [rest]) if g.geom_type == "Polygon" and g.area / pt2_per_m2 >= 4.0]
            else:
                new.append((poly, r))
        rooms_poly = new
        log(f"      open-plan split: {n_open} large regions with >= {osp.get('min_seeds', 2)} room labels split from their labels -> {len(rooms_poly)} rooms")

    sr = cfg.get("structural_pass", {})
    if sr.get("enabled") and texts:
        th = rasterize_thick(P, skip, roi, S)
        res = structural_rooms(th, chords_full, texts, sr["keywords"], roi, S, pt2_per_m2, [p for p, _ in rooms_poly], tuple(sr.get("area_m2", (1.0, 60.0))))
        drop = {i for _, rep, _ in res for i in rep}
        base = dict(label=-1, area_pt2=0, bbox=(0, 0, 0, 0), border=False, inscribed_pt=0.0)
        rooms_poly = [rp for i, rp in enumerate(rooms_poly) if i not in drop] + [(pg, dict(base, structural=txt, area_pt2=pg.area, inscribed_pt=7.0)) for pg, _, txt in res]
        log(f"      structural pass: {len(res)} named spaces re-segmented ({', '.join(t for *_, t in res)}), replaced {len(drop)} fragments")

    tick("structural_pass")
    pods = []
    fx = cfg.get("fixtures", {})
    if fx.get("pod_exemplars"):
        L = line_layer(P, roi, fx.get("px_per_pt", 3), skip=grid)
        for ex in fx["pod_exemplars"]:
            for m in find_symbol(L, roi, fx.get("px_per_pt", 3), ex, list(V.get("footprint") or [list(foot.bounds)]) + list(fx.get("extra_search_boxes", [])), thresh=fx.get("thresh", 0.85), device=dev):
                if all(np.linalg.norm(np.array(m["center_pt"]) - np.array(q["center_pt"])) > 20 for q in pods):
                    pods.append(m)
        log(f"      sanitary-pod symbol matches={len(pods)}")
        new, used_pod = [], set()
        for poly, r in rooms_poly:
            mine = [(i, p) for i, p in enumerate(pods) if i not in used_pod and poly.contains(Point(p["center_pt"]))]
            whole_is_pod = False
            for i, p in mine:
                used_pod.add(i)
                pb = box(*p["box_pt"])
                if poly.area <= 1.6 * pb.area:      # region is already isolated as the pod itself
                    whole_is_pod = True
                    new.append((poly, dict(r, forced="bathroom", pod=p)))
                    continue
                cut, rest = poly.intersection(pb), poly.difference(pb)
                if cut.is_empty or rest.is_empty or cut.area < 0.25 * pb.area:
                    continue                         # pod box barely overlaps this region: leave the region untouched
                cut = cut if cut.geom_type == "Polygon" else max(cut.geoms, key=lambda g: g.area)
                new.append((cut, dict(r, forced="bathroom", pod=p)))
                poly = rest
            if not whole_is_pod:
                # a pod cut can disconnect a region: keep every remaining piece, not only the largest
                parts = [poly] if poly.geom_type == "Polygon" else sorted(poly.geoms, key=lambda g: -g.area)
                for g in parts:
                    if g.area / pt2_per_m2 >= 0.3:
                        new.append((g, dict(r, ensuite_pods=len(mine))))
        # pods outside every room polygon (e.g. the truncated annex): standalone bathroom, explicitly flagged (no host room)
        base_r = dict(label=-1, area_pt2=0, bbox=(0, 0, 0, 0), border=False, inscribed_pt=7.0)
        for i, p in enumerate(pods):
            if i not in used_pod:
                new.append((box(*p["box_pt"]), dict(base_r, forced="bathroom", pod=p, standalone_pod=True)))
        rooms_poly = new

    tick("pod_template_matching")
    log("[5/6] classification")
    rooms = []
    circle_rooms = []
    for poly, r in rooms_poly:
        inside = [t["text"] for t in texts
                  if poly.contains(Point((t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2))]
        if r.get("open_core"):                       # the open space itself: room labels that happen to lie in it belong to the rooms around it
            inside = [t for t in inside if not match_keywords(t, {n: sp for n, sp in classes.items() if n not in (
                "corridor_or_hallway", "communal_or_shared_space", "unknown", "other", "reception_or_entrance")})[0]]
        if not inside and describe(poly)["shape_type"] == "circle" and poly.area / pt2_per_m2 < 8:
            circle_rooms.append(poly)                # a closed round outline with no label (tree, grid bubble, column): a circle, not a room
            continue
        nc = sum(poly.contains(Point(c)) for c, _ in circles)
        info = classify_room(poly, inside, nc, classes, pt2_per_m2)
        if r.get("forced"):
            info.update(type=r["forced"], heuristic_score=r["pod"]["score"], evidence=[f"symbol_template_match:{r['pod']['score']}"],
                        review_required=True, flags=info["flags"] + ["boundary_is_symbol_bounding_box"]
                        + (["no_host_room:drawing_truncated_here"] if r.get("standalone_pod") else []))
        rec = dict(id="", type=info.pop("type"), **polygon_to_record(poly, scale_r), shape=describe(poly))
        rec.update(confidence=None, confidence_source="none: rule-based, no calibrated probability",
                   heuristic_score=info.pop("heuristic_score"), source="vector+ocr",
                   inscribed_radius_pt=round(r["inscribed_pt"], 1), text_inside=inside, **info)
        if r.get("split_from"):
            rec["flags"] = rec["flags"] + [f"split_by_label_watershed:{r['n_split']}"]
            rec["review_required"] = True
        rooms.append(rec)
    # adopt small unlabelled pockets into the bedroom or bathroom they share the longest boundary with (parts of the same unit missed by the pod template box)
    ad = cfg.get("adoption", {})
    n_adopted = 0
    if ad.get("enabled"):
        for r in list(rooms):
            if r["type"] != "unknown" or r["text_inside"] or r["approx_area_m2"] >= ad.get("max_area_m2", 4.0):
                continue
            pg = Polygon(r["polygon_pt"])
            best = None
            for host in rooms:
                if host["type"] not in ("bedroom", "bathroom"):
                    continue
                shared = pg.buffer(0.8).intersection(Polygon(host["polygon_pt"])).length / 2
                if shared >= ad.get("min_shared_pt", 8.0) and (best is None or shared > best[0]):
                    best = (shared, host)
            if best:
                h = best[1]
                h.setdefault("extra_polygons", []).append(r["polygon"])
                h.setdefault("extra_polygons_pt", []).append(r["polygon_pt"])
                h["approx_area_m2"] = round(h["approx_area_m2"] + r["approx_area_m2"], 1)
                h["area_pixels"] = round(h["area_pixels"] + r["area_pixels"], 1)
                h["flags"] = h["flags"] + [f"adopted_fragment:{r['approx_area_m2']}m2"]
                h["review_required"] = True
                rooms.remove(r)
                n_adopted += 1
        log(f"      adopted {n_adopted} unlabelled pockets into neighbouring bedrooms/bathrooms")

    cr = cfg.get("circulation_rule", {})
    n_circ = 0
    if cr.get("enabled"):
        circ = [Polygon(r["polygon_pt"]) for r in rooms if r["type"] in ("corridor_or_hallway", "communal_or_shared_space")]
        for r in rooms:
            if r["type"] == "unknown" and not r["text_inside"] and cr["min_area_m2"] <= r["approx_area_m2"] <= cr["max_area_m2"]:
                pg = Polygon(r["polygon_pt"])
                if any(pg.distance(c) <= cr.get("max_gap_pt", 2.0) for c in circ):
                    r.update(type="corridor_or_hallway", heuristic_score=0.4, evidence=r["evidence"] + ["unlabelled_adjacent_to_corridor"], review_required=True)
                    r["flags"] = r["flags"] + ["type_by_adjacency_rule_unverified"]
                    n_circ += 1
        log(f"      circulation rule re-typed {n_circ} unlabelled lobbies")

    min_a = V.get("min_room_area_m2", 0)
    frags = [r for r in rooms if r["approx_area_m2"] < min_a and r["type"] != "bathroom"]
    rooms = [r for r in rooms if r not in frags]
    write_json(out / "debug" / "discarded_fragments.json", [dict(polygon_pt=r["polygon_pt"], approx_area_m2=r["approx_area_m2"]) for r in frags])
    log(f"      discarded {len(frags)} fragments < {min_a} m2")
    rooms.sort(key=lambda r: (np.mean(np.array(r["polygon_pt"])[:, 1]) // 150, np.mean(np.array(r["polygon_pt"])[:, 0])))
    for i, r in enumerate(rooms, 1):
        r["id"] = f"room-{i:04d}"

    tick("classification_adoption")
    wcfg = cfg.get("windows", {})
    if wcfg.get("exemplars"):
        wl = line_layer(P, roi, wcfg.get("px_per_pt", 3), w_range=tuple(wcfg.get("w_range", (0.4, 0.56))))
        windows = window_elements_template(wl, roi, wcfg.get("px_per_pt", 3), wcfg["exemplars"],
                                           [(b[0] - 20, b[1] - 20, b[2] + 20, b[3] + 20) for b in (V.get("footprint") or [list(foot.bounds)])], foot,
                                           wcfg.get("thresh", 0.75), wcfg.get("band_pt", 16.0), device=dev)
    else:
        windows = window_elements(P, foot)
    log(f"      windows detected={len(windows)}")
    circle_els = circle_elements(P, f, min_wall, roi)
    log(f"      circles={len(circle_els)} (+{len(circle_rooms)} round unlabelled regions kept out of the room list)")
    elements = door_elements(arcs, chords) + column_elements(P, foot, f) + windows + circle_els
    # wall polygons from the closed wall mask restricted to the footprint
    fm = np.zeros_like(closed)
    for g in getattr(foot, "geoms", [foot]):
        if g.geom_type == "Polygon":
            cv2.fillPoly(fm, [((np.array(g.exterior.coords) - roi[:2]) * S).astype(np.int32)], 255)
    cnts, hier = cv2.findContours(cv2.bitwise_and(closed, fm), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1 or len(c) < 3 or cv2.contourArea(c) / S ** 2 < 8:
            continue
        holes = [h[:, 0, :] / S + roi[:2] for j, h in enumerate(cnts)
                 if hier[0][j][3] == i and len(h) >= 3 and cv2.contourArea(h) / S ** 2 >= 20]
        pg = Polygon(c[:, 0, :] / S + roi[:2], holes)
        if not pg.is_valid:                      # holes touching the shell etc.: repair rather than drop
            pg = pg.buffer(0)
        for part in ([pg] if pg.geom_type == "Polygon" else [g for g in pg.geoms if g.geom_type == "Polygon"]):
            if part.is_empty or part.area < 8:
                continue
            part = part.simplify(0.5)
            elements.append(dict(type="wall", polygon_pt=np.array(part.exterior.coords)[:-1].round(2).tolist(),
                                 holes_pt=[np.array(r.coords)[:-1].round(2).tolist() for r in part.interiors],
                                 source="vector:stroke_mask"))

    tick("elements_walls")
    log("[6/6] writing outputs")
    base = cv2.imdecode(np.frombuffer(page.get_pixmap(matrix=pymupdf.Matrix(scale_r, scale_r)).tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    H, W = base.shape[:2]
    full_pt = [0.0, 0.0, W / scale_r, H / scale_r]          # output canvas = the ENTIRE page (never the detection ROI)
    zoom_pt = [max(roi[0] - 10, 0), max(roi[1] - 10, 0), roi[2] + 10, roi[3] + 10]
    used = {}
    for r in rooms:
        used[r["type"]] = used.get(r["type"], 0) + 1
    extra = [("red ring = review required", (0, 0, 255)), ("red dot = door hinge", (0, 0, 220)), ("blue box = window", (230, 160, 0)), ("grey box = column", (60, 60, 60))]
    ov_full = draw_overlay(base, rooms, classes, elements, full_pt, scale_r)
    assert ov_full.shape == base.shape, "overlay must have exactly the page render's size"
    ov_full, legend_info = add_legend_inset(ov_full, classes, used, extra)
    cv2.imwrite(str(out / "overlay.png"), ov_full)
    # clearly-named zoom crop of the detection ROI (extra view, not the primary output)
    zx0, zy0, zx1, zy1 = [int(v * scale_r) for v in zoom_pt]
    cv2.imwrite(str(out / "overlay_plan_zoom_cropped.png"), ov_full[zy0:zy1, zx0:zx1])
    cv2.imwrite(str(out / "debug" / "render_reference.png"), base)
    mk = np.full(base.shape, 255, np.uint8)
    for r in rooms:
        colr = tuple(int(c) for c in classes[r["type"]]["color"][::-1])
        cv2.fillPoly(mk, [np.array(r["polygon"]).astype(np.int32)], colr)
        for ex in r.get("extra_polygons", []):
            cv2.fillPoly(mk, [np.array(ex).astype(np.int32)], colr)
        for h in r["holes"]:
            cv2.fillPoly(mk, [np.array(h).astype(np.int32)], (255, 255, 255))
    assert mk.shape == base.shape
    cv2.imwrite(str(out / "room_masks.png"), mk)
    cv2.imwrite(str(out / "debug" / "wall_mask_roi.png"), cv2.resize(255 - closed, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))

    image = dict(width=W, height=H, coordinate_origin="top_left", units="pixels",
                 output_canvas="full_page", detection_roi_pt=list(roi), building_footprint_pt=[list(b) for b in V["footprint"]] if V.get("footprint") else [np.array(g.exterior.coords).round(1).tolist() for g in getattr(foot, "geoms", [foot])],
                 calibration=cal,
                 legend=legend_info,
                 transform=dict(type="pdf_page_render", scale_px_per_pt=scale_r, page_rotation_applied_deg=page.rotation,
                                page_size_pt=[page.rect.width, page.rect.height], pt_to_px="px = pt * scale_px_per_pt"),
                 note="Pixel coordinates refer to the page rendered at scale_px_per_pt (outputs/debug/render_reference.png). "
                      "approx_area_m2 assumes the printed scale 1:100 (read from the title block, A1 sheet) and is only approximate.")
    write_json(out / "rooms.json", dict(image=image, rooms=rooms))
    write_json(out / "elements.json", dict(image=image, elements=elements))
    tick("write_outputs")
    if sampler:
        sampler._stop.set()
    stats = dict(device=dev, device_info=dev_info, ocr_providers=(ocr_info.get("ocr_providers") if cfg["ocr"]["enabled"] else None),
                 gpu_measured=(sampler.summary() if sampler else None), stage_times_s=T, runtime_s=round(time.time() - t0, 1), n_rooms=len(rooms), by_type=used,
                 n_elements={k: sum(e["type"] == k for e in elements) for k in ("door", "window", "column", "wall")},
                 peak_ram_mb=_peak_ram_mb(), note="neural network inference is limited to the RapidOCR det/cls/rec models (ONNX Runtime); everything else is classical geometry")
    write_json(out / "run_stats.json", stats)
    log(f"done in {stats['runtime_s']}s: {stats}")
    return rooms, elements, stats
