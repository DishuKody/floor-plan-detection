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
from .vector.fixtures_template import line_layer, find_symbol
from .vector.elements import door_elements, column_elements, window_elements, circle_fixtures
from .postprocessing.polygons import region_polygon, polygon_to_record
from .postprocessing.classify import classify_room, UNIT_ID
from .postprocessing.instances import split_region
from .postprocessing.structural import structural_rooms
from .output.visualization import draw_overlay, add_legend
from .output.json_writer import write_json


def _peak_ram_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().peak_wset / 2 ** 20)
    except Exception:
        return None


def run_vector(pdf, cfg, classes, out_dir, log=print):
    t0 = time.time()
    out = Path(out_dir)
    (out / "debug").mkdir(parents=True, exist_ok=True)
    V = cfg["vector"]
    scale_r = cfg["input"]["render_scale"]
    S = V["raster_px_per_pt"]
    doc = pymupdf.open(pdf)
    page = doc[0]
    roi = tuple(V["roi"])
    # pt^2 per real m^2 at the printed drawing scale (1 pt = 25.4/72 mm on paper)
    pt2_per_m2 = (1000.0 / (25.4 / 72 * cfg["drawing"]["scale_ratio"])) ** 2
    foot = unary_union([box(*b) for b in V["footprint"]])

    log("[1/6] extracting vector paths")
    P = extract_paths(page)
    if len(P) < 500:
        raise RuntimeError(
            f"{Path(pdf).name}: only {len(P)} vector paths ({len(page.get_images())} embedded raster image(s)). This is a raster-only "
            "drawing; the vector pipeline cannot be used. A raster model (e.g. Raster2Seq / segmentation) is required - see analysis_report.md.")
    grid = find_grid_paths(P)
    da = V["door_arc"]
    arcs = detect_door_arcs(P, r_range=(da["r_min"], da["r_max"]), sweep_range=(da["sweep_min"], da["sweep_max"]))
    log(f"      paths={len(P)} grid_dash_paths_removed={len(grid)} door_arcs={len(arcs)}")

    log("[2/6] wall mask, wall-end extension, door-opening closure")
    skip = grid | {j for a in arcs for j in a.path_indices}
    if V.get("remove_tread_series", False):
        treads = find_tread_series(P, skip)
        skip |= treads
        log(f"      stair-tread/hatch series removed from wall mask: {len(treads)} segments")
    mask = rasterize_mask(P, skip, roi, S, min_w=V["min_wall_stroke"])
    ext = extend_dangling_walls(P, skip, mask, roi, S, **V["extend_walls"])
    chords_full = close_doors(mask, arcs, roi, S)
    arcs = [c[3] for c in chords_full]            # arcs accepted as doors (thin arcs need wall evidence)
    chords = [c[:3] for c in chords_full]
    log(f"      doors accepted={len(arcs)}")
    closed, lab, regs = label_regions(mask, S, V["wall_channel_close_pt"], V["min_area_pt2"], V["min_inscribed_radius_pt"])
    log(f"      wall extensions={len(ext)} candidate regions={len(regs)}")

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

    texts = []
    if cfg["ocr"]["enabled"]:
        log("[4/6] OCR (local RapidOCR/ONNX)")
        from .supporting_signals.ocr import run_ocr
        texts = run_ocr(page, roi, cfg["ocr"])
        write_json(out / "debug" / "ocr.json", texts)
        log(f"      text boxes={len(texts)}")
    circles = circle_fixtures(P)

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

    sr = cfg.get("structural_pass", {})
    if sr.get("enabled") and texts:
        th = rasterize_thick(P, skip, roi, S)
        res = structural_rooms(th, chords_full, texts, sr["keywords"], roi, S, pt2_per_m2, [p for p, _ in rooms_poly], tuple(sr.get("area_m2", (1.0, 60.0))))
        drop = {i for _, rep, _ in res for i in rep}
        base = dict(label=-1, area_pt2=0, bbox=(0, 0, 0, 0), border=False, inscribed_pt=0.0)
        rooms_poly = [rp for i, rp in enumerate(rooms_poly) if i not in drop] + [(pg, dict(base, structural=txt, area_pt2=pg.area, inscribed_pt=7.0)) for pg, _, txt in res]
        log(f"      structural pass: {len(res)} named spaces re-segmented ({', '.join(t for *_, t in res)}), replaced {len(drop)} fragments")

    pods = []
    fx = cfg.get("fixtures", {})
    if fx.get("pod_exemplars"):
        L = line_layer(P, roi, fx.get("px_per_pt", 3), skip=grid)
        for ex in fx["pod_exemplars"]:
            for m in find_symbol(L, roi, fx.get("px_per_pt", 3), ex, V["footprint"], thresh=fx.get("thresh", 0.85)):
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
        rooms_poly = new

    log("[5/6] classification")
    rooms = []
    for poly, r in rooms_poly:
        inside = [t["text"] for t in texts
                  if poly.contains(Point((t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2))]
        nc = sum(poly.contains(Point(c)) for c, _ in circles)
        info = classify_room(poly, inside, nc, classes, pt2_per_m2)
        if r.get("forced"):
            info.update(type=r["forced"], heuristic_score=r["pod"]["score"], evidence=[f"symbol_template_match:{r['pod']['score']}"],
                        review_required=True, flags=info["flags"] + ["boundary_is_symbol_bounding_box"])
        rec = dict(id="", type=info.pop("type"), **polygon_to_record(poly, scale_r))
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

    min_a = V.get("min_room_area_m2", 0)
    frags = [r for r in rooms if r["approx_area_m2"] < min_a and r["type"] != "bathroom"]
    rooms = [r for r in rooms if r not in frags]
    write_json(out / "debug" / "discarded_fragments.json", [dict(polygon_pt=r["polygon_pt"], approx_area_m2=r["approx_area_m2"]) for r in frags])
    log(f"      discarded {len(frags)} fragments < {min_a} m2")
    rooms.sort(key=lambda r: (np.mean(np.array(r["polygon_pt"])[:, 1]) // 150, np.mean(np.array(r["polygon_pt"])[:, 0])))
    for i, r in enumerate(rooms, 1):
        r["id"] = f"room-{i:04d}"

    elements = door_elements(arcs, chords) + column_elements(P, foot) + window_elements(P, foot)
    # wall polygons from the closed wall mask restricted to the footprint
    fm = np.zeros_like(closed)
    for b in V["footprint"]:
        x0, y0, x1, y1 = [int(v * S) for v in (b[0] - roi[0], b[1] - roi[1], b[2] - roi[0], b[3] - roi[1])]
        fm[max(y0, 0):y1, max(x0, 0):x1] = 255
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

    log("[6/6] writing outputs")
    base = cv2.imdecode(np.frombuffer(page.get_pixmap(matrix=pymupdf.Matrix(scale_r, scale_r)).tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    H, W = base.shape[:2]
    crop = [max(roi[0] - 10, 0), max(roi[1] - 10, 0), roi[2] + 10, roi[3] + 10]
    used = {}
    for r in rooms:
        used[r["type"]] = used.get(r["type"], 0) + 1
    ov = draw_overlay(base, rooms, classes, elements, crop, scale_r)
    cv2.imwrite(str(out / "overlay.png"), add_legend(ov, classes, used))
    cv2.imwrite(str(out / "debug" / "render_reference.png"), base)
    mk = np.full(base.shape, 255, np.uint8)
    for r in rooms:
        cv2.fillPoly(mk, [np.array(r["polygon"]).astype(np.int32)], tuple(int(c) for c in classes[r["type"]]["color"][::-1]))
        for ex in r.get("extra_polygons", []):
            cv2.fillPoly(mk, [np.array(ex).astype(np.int32)], tuple(int(c) for c in classes[r["type"]]["color"][::-1]))
        for h in r["holes"]:
            cv2.fillPoly(mk, [np.array(h).astype(np.int32)], (255, 255, 255))
    x0, y0, x1, y1 = [int(v * scale_r) for v in crop]
    cv2.imwrite(str(out / "room_masks.png"), mk[y0:y1, x0:x1])
    cv2.imwrite(str(out / "debug" / "wall_mask.png"), cv2.resize(255 - closed, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))

    image = dict(width=W, height=H, coordinate_origin="top_left", units="pixels",
                 transform=dict(type="pdf_page_render", scale_px_per_pt=scale_r, page_rotation_applied_deg=page.rotation,
                                page_size_pt=[page.rect.width, page.rect.height], pt_to_px="px = pt * scale_px_per_pt"),
                 note="Pixel coordinates refer to the page rendered at scale_px_per_pt (outputs/debug/render_reference.png). "
                      "approx_area_m2 assumes the printed scale 1:100 (read from the title block, A1 sheet) and is only approximate.")
    write_json(out / "rooms.json", dict(image=image, rooms=rooms))
    write_json(out / "elements.json", dict(image=image, elements=elements))
    stats = dict(runtime_s=round(time.time() - t0, 1), n_rooms=len(rooms), by_type=used,
                 n_elements={k: sum(e["type"] == k for e in elements) for k in ("door", "window", "column", "wall")},
                 peak_ram_mb=_peak_ram_mb(), gpu_used=False, note="vector path is CPU-only; no neural network inference except CPU OCR (RapidOCR/ONNX)")
    write_json(out / "run_stats.json", stats)
    log(f"done in {stats['runtime_s']}s: {stats}")
    return rooms, elements, stats
