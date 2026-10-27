"""Usage: python -m src.evaluation.compare_versions <dir_A> <dir_B> <out_dir>
Compares two output directories (A = old, B = new) with the same metrics and writes comparison.json, comparison.md and
side-by-side overlay panels. Every number is recomputed from rooms.json / ocr.json / images, not copied."""
import json
import sys
from pathlib import Path
import numpy as np
import cv2
from shapely.geometry import Polygon, LineString
from .unit_id_proxy import evaluate
from .named_spaces import evaluate_named
from .bedroom_consistency import evaluate_bedrooms
from .report import geometry_checks

ROI = (40, 330)
PANELS = {   # name: (x0, y0, x1, y1) in page points
    "core_lobby_stair_riser_aov": (690, 860, 1130, 1160),
    "wing_5bed_cluster": (725, 360, 1135, 700),
    "wing_pod_pockets_closeup": (735, 470, 1030, 740),
    "east_end": (1380, 860, 1720, 1270),
}


def boundary_fit(rooms, wall_png):
    wm = cv2.imread(str(wall_png), 0) < 128          # 2 px/pt, origin = ROI
    K = 2.0
    dt = cv2.distanceTransform((~wm).astype(np.uint8), cv2.DIST_L2, 3) / K
    off, ink = [], []
    for r in rooms:
        pg = Polygon(r["polygon_pt"])
        ls = LineString(list(pg.exterior.coords))
        n = max(int(ls.length), 8)
        pts = np.array([ls.interpolate(i / n, normalized=True).coords[0] for i in range(n)])
        px = ((pts - ROI) * K).astype(int)
        px[:, 0] = px[:, 0].clip(0, wm.shape[1] - 1); px[:, 1] = px[:, 1].clip(0, wm.shape[0] - 1)
        off.append(float((dt[px[:, 1], px[:, 0]] > 3).mean()))
        m = np.zeros(wm.shape, np.uint8)
        cv2.fillPoly(m, [((np.array(r["polygon_pt"]) - ROI) * K).astype(np.int32)], 1)
        ink.append(float((wm & (m > 0)).sum()) / max(m.sum(), 1))
    return dict(mean_frac_boundary_gt3pt_from_wall=round(float(np.mean(off)), 3), max_wall_ink_fraction_inside=round(float(np.max(ink)), 4))


def coord_check(rooms, render_png, scale):
    img = cv2.imread(str(render_png), 0)
    dt = cv2.distanceTransform((img >= 128).astype(np.uint8), cv2.DIST_L2, 3)    # distance to nearest ink pixel
    med = []
    for r in rooms:
        ls = LineString(r["polygon"])
        n = int(ls.length / 3) + 4
        p = np.array([ls.interpolate(i / n, normalized=True).coords[0] for i in range(n)]).astype(int)
        p[:, 0] = p[:, 0].clip(0, img.shape[1] - 1); p[:, 1] = p[:, 1].clip(0, img.shape[0] - 1)
        med.append(float(np.median(dt[p[:, 1], p[:, 0]])))
    mx = max(np.abs(np.array(r["polygon"]) - np.array(r["polygon_pt"]) * scale).max() for r in rooms)
    return dict(mean_median_px_boundary_to_ink=round(float(np.mean(med)), 2), max_abs_px_vs_pt_x_scale=round(float(mx), 3))


def collect(d):
    d = Path(d)
    J = json.load(open(d / "rooms.json"))
    rooms, ocr = J["rooms"], json.load(open(d / "debug" / "ocr.json"))
    ev = evaluate(rooms, ocr)
    named = evaluate_named(rooms, ocr)
    bed = evaluate_bedrooms(rooms)
    stats = json.load(open(d / "run_stats.json"))
    el = json.load(open(d / "elements.json"))["elements"]
    types = {}
    for r in rooms:
        types[r["type"]] = types.get(r["type"], 0) + 1
    big = max(rooms, key=lambda r: r["approx_area_m2"])
    return dict(
        n_rooms=len(rooms), by_type=dict(sorted(types.items())), n_review_required=sum(r["review_required"] for r in rooms),
        n_unknown=types.get("unknown", 0), largest_room=dict(id=big["id"], type=big["type"], m2=big["approx_area_m2"]),
        unit_id_proxy={k: dict(n=v["n_units"], isolated_correct=v["isolated_and_correct"], counts=v["counts"]) for k, v in ev["summary"].items()},
        named_spaces=dict(ok=named["n_ok"], checked=named["n_checked"], failing={k: v["issues"] for k, v in named["spaces"].items() if not v["ok"]},
                          passing=sorted(k for k, v in named["spaces"].items() if v["ok"])),
        bedroom_consistency={k: dict(n=v["n"], cv=v["cv"], outliers_gt10pct=v["n_outliers_gt_10pct"]) for k, v in bed["groups"].items()},
        pockets_touching_bedroom=bed["pockets_touching_bedroom"],
        geometry=geometry_checks(rooms), boundary_fit=boundary_fit(rooms, d / "debug" / "wall_mask.png"),
        coordinates=coord_check(rooms, d / "debug" / "render_reference.png", J["image"]["transform"]["scale_px_per_pt"]),
        elements={k: sum(e["type"] == k for e in el) for k in ("door", "window", "column", "wall")},
        runtime_s=stats["runtime_s"], peak_ram_mb=stats.get("peak_ram_mb"))


def panels(a, b, out):
    ia, ib = cv2.imread(str(Path(a) / "overlay.png")), cv2.imread(str(Path(b) / "overlay.png"))
    names = []
    for name, (x0, y0, x1, y1) in PANELS.items():
        sl = (slice(int((y0 - 320) * 1.5), int((y1 - 320) * 1.5)), slice(int((x0 - 30) * 1.5), int((x1 - 30) * 1.5)))
        ca, cb = ia[sl].copy(), ib[sl].copy()
        h = max(ca.shape[0], cb.shape[0])
        pad = lambda c: cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for c, t in ((ca, "v1 (before)"), (cb, "v2 (after)")):
            cv2.rectangle(c, (0, 0), (190, 30), (255, 255, 255), -1)
            cv2.putText(c, t, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        panel = np.hstack([pad(ca), np.full((h, 8, 3), 0, np.uint8), pad(cb)])
        f = Path(out) / f"{name}.png"
        cv2.imwrite(str(f), panel)
        names.append(f.name)
    return names


def main(a, b, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    A, B = collect(a), collect(b)
    imgs = panels(a, b, out)
    json.dump(dict(v1=A, v2=B, panels=imgs), open(out / "comparison.json", "w"), indent=1)
    row = lambda k, f: f"| {k} | {f(A)} | {f(B)} |"
    L = ["| metric | v1 | v2 |", "|---|---|---|",
         row("rooms", lambda x: x["n_rooms"]), row("by type", lambda x: ", ".join(f"{k}:{v}" for k, v in x["by_type"].items())),
         row("unknown rooms", lambda x: x["n_unknown"]), row("flagged review_required", lambda x: x["n_review_required"]),
         row("largest room", lambda x: f"{x['largest_room']['id']} {x['largest_room']['type']} {x['largest_room']['m2']} m2"),
         row("bedroom units isolated+correct (OCR unit-ID proxy)", lambda x: f"{x['unit_id_proxy']['bedroom']['isolated_correct']}/{x['unit_id_proxy']['bedroom']['n']}"),
         row("kitchen units isolated+correct", lambda x: f"{x['unit_id_proxy']['kitchen']['isolated_correct']}/{x['unit_id_proxy']['kitchen']['n']}"),
         row("named spaces OK (stair, riser, lifts, AOV, store, lobby, tutor flat)", lambda x: f"{x['named_spaces']['ok']}/{x['named_spaces']['checked']}"),
         row("named spaces failing", lambda x: "; ".join(f"{k}:{','.join(v)}" for k, v in x["named_spaces"]["failing"].items()) or "-"),
         row("B503 bedroom area CV / outliers", lambda x: f"{x['bedroom_consistency'].get('TYPEB503', {}).get('cv')} / {x['bedroom_consistency'].get('TYPEB503', {}).get('outliers_gt10pct')}"),
         row("B703 bedroom area CV / outliers", lambda x: f"{x['bedroom_consistency'].get('TYPEB703', {}).get('cv')} / {x['bedroom_consistency'].get('TYPEB703', {}).get('outliers_gt10pct')}"),
         row("unknown pockets touching a bedroom", lambda x: x["pockets_touching_bedroom"]),
         row("invalid polygons / overlapping pairs", lambda x: f"{x['geometry']['invalid_polygons']} / {x['geometry']['pairs_overlapping_gt_2pct']}"),
         row("boundary >3pt from a wall (mean fraction)", lambda x: x["boundary_fit"]["mean_frac_boundary_gt3pt_from_wall"]),
         row("max wall-ink fraction inside a polygon", lambda x: x["boundary_fit"]["max_wall_ink_fraction_inside"]),
         row("coordinate check: boundary-to-ink px / max |px - pt*s|", lambda x: f"{x['coordinates']['mean_median_px_boundary_to_ink']} / {x['coordinates']['max_abs_px_vs_pt_x_scale']}"),
         row("elements door/window/column/wall", lambda x: "/".join(str(x["elements"][k]) for k in ("door", "window", "column", "wall"))),
         row("runtime s / peak RAM MB", lambda x: f"{x['runtime_s']} / {x['peak_ram_mb']}")]
    open(out / "comparison.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:4])
