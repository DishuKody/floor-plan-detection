"""Reference-free quality proxies for a vector-mode run (no manual ground truth exists for these sheets, so these are PROXIES, not accuracy).
Usage: python -m src.evaluation.vector_proxies <pdf> <outputs_dir> [--scale 200] [--min-stroke 0.315] [--json out.json]

  label_isolation   a printed room name is a free, independent signal: the share of named-room labels whose room holds no other room's name
                    (1.0 = every named room is separated from its neighbours; low = rooms merged)
  merged_rooms      rooms that hold >= 2 different named-room labels (words of one label are clustered first)
  boundary_on_wall  share of room-boundary length lying within `tol_pt` of a structural wall stroke (the rest is door chords, wall-end
                    extensions, open-plan cuts: virtual boundaries)
  large_rooms       rooms above 150 m2 (merged concourse / site-area leaks)
  false_fixtures    bathrooms that contain no wall-bounded fixture evidence are not judged here; trees / grid bubbles show up as rooms of
                    shape 'circle' and as 'bathroom' via the round-fixture rule, both counted
"""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
import pymupdf
import yaml
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[2]


def named_labels(texts, classes, f, cluster_m=4.0):
    from ..postprocessing.classify import match_keywords
    room_cls = {n: s for n, s in classes.items() if n not in ("corridor_or_hallway", "communal_or_shared_space", "unknown", "other", "reception_or_entrance")}
    pts = []
    for t in texts:
        if not match_keywords(t["text"], room_cls)[0]:
            continue
        c = np.array([(t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2])
        for q in pts:
            if np.linalg.norm(q[0] / q[1] - c) < cluster_m * 28.346 * f:
                q[0] += c; q[1] += 1; break
        else:
            pts.append([c.copy(), 1])
    return [q[0] / q[1] for q in pts]


def boundary_on_wall(pdf, rooms, f, min_stroke, tol_pt=2.0):
    from ..vector.extract_pdf_paths import extract_paths
    page = pymupdf.open(pdf)[0]
    P = extract_paths(page)
    s = 2.0
    m = np.zeros((int(page.rect.height * s) + 1, int(page.rect.width * s) + 1), np.uint8)
    for q in P:
        if q.width >= min_stroke and not q.dashed:
            cv2.polylines(m, [(q.pts * s).astype(np.int32)], q.closed, 255, 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * tol_pt * s) | 1,) * 2))
    on = tot = 0.0
    for r in rooms:
        ring = np.array(r["polygon_pt"])
        for a, b in zip(ring, np.roll(ring, -1, axis=0)):
            L = float(np.linalg.norm(b - a))
            n = max(2, int(L))
            xy = (a + (b - a) * np.linspace(0, 1, n)[:, None]) * s
            xi = np.clip(xy[:, 0].astype(int), 0, m.shape[1] - 1)
            yi = np.clip(xy[:, 1].astype(int), 0, m.shape[0] - 1)
            on += L * float((m[yi, xi] > 0).mean())
            tot += L
    return on / max(tot, 1e-9)


def proxies(pdf, out_dir, ratio, min_stroke, classes):
    out = Path(out_dir)
    R = json.load(open(out / "rooms.json"))["rooms"]
    E = json.load(open(out / "elements.json"))["elements"]
    texts = json.load(open(out / "debug" / "ocr.json"))
    f = 100.0 / ratio
    seeds = named_labels(texts, classes, f)
    polys = [Polygon(r["polygon_pt"]) for r in R]
    tree = STRtree(polys)
    holders = {}
    for i, sd in enumerate(seeds):
        for j in tree.query(Point(sd)):
            if polys[j].contains(Point(sd)):
                holders.setdefault(int(j), []).append(i)
    isolated = sum(1 for js in holders.values() for _ in js if len(js) == 1)
    n_lab = sum(len(v) for v in holders.values())
    cnt = {}
    for r in R:
        cnt[r["type"]] = cnt.get(r["type"], 0) + 1
    return dict(n_rooms=len(R), by_type=cnt, named_labels=len(seeds), named_labels_inside_a_room=n_lab,
                label_isolation=round(isolated / max(n_lab, 1), 3), merged_rooms=sum(1 for v in holders.values() if len(v) >= 2),
                rooms_with_named_label=len(holders), large_rooms_over_150m2=sum(r["approx_area_m2"] > 150 for r in R),
                max_room_m2=max(r["approx_area_m2"] for r in R), circle_shaped_rooms=sum(r["shape"]["shape_type"] == "circle" for r in R),
                unknown_share=round(cnt.get("unknown", 0) / max(len(R), 1), 3),
                boundary_on_wall=round(boundary_on_wall(pdf, R, f, min_stroke), 3),
                elements={k: sum(e["type"] == k for e in E) for k in sorted({e["type"] for e in E})})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf"); ap.add_argument("out_dir")
    ap.add_argument("--scale", type=float, default=100.0); ap.add_argument("--min-stroke", type=float, default=0.2)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    classes = yaml.safe_load(open(ROOT / "config" / "classes.yaml"))["classes"]
    res = proxies(a.pdf, a.out_dir, a.scale, a.min_stroke, classes)
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
