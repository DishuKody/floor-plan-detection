"""Proxy evaluation from the drawing's own unit-ID labels (e.g. 02/D3/12/7). NOT manual ground truth:
it measures room-count errors (merged / missed / mislabelled) for labelled units only; it says nothing about boundary accuracy."""
import re
import json
from collections import defaultdict
import numpy as np
from shapely.geometry import Polygon, Point

UNIT = re.compile(r"\b0\d/D\d/\d+/(\d+|K|TU)\b", re.I)


def unit_labels(ocr):
    """-> {unit_id: (x, y)} centre of each full unit-ID read (partial reads like '2/3' are ignored)."""
    pos = defaultdict(list)
    for t in ocr:
        m = UNIT.search(t["text"])
        if m:
            b = t["bbox"]
            pos[m.group(0).upper()].append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    return {k: tuple(np.mean(v, axis=0)) for k, v in pos.items()}


def expected_type(unit_id):
    last = unit_id.split("/")[-1]
    return "kitchen" if last == "K" else ("other" if last == "TU" else "bedroom")


def evaluate(rooms, ocr):
    polys = [(r["id"], r["type"], Polygon(r["polygon_pt"])) for r in rooms]
    units = unit_labels(ocr)
    res = defaultdict(list)
    per_unit = {}
    for uid, (x, y) in sorted(units.items()):
        p = Point(x, y)
        hit = [(rid, typ) for rid, typ, pg in polys if pg.contains(p)]
        exp = expected_type(uid)
        if not hit:
            # label may sit on a wall/pod boundary: look for the nearest room within 6pt
            near = sorted((pg.distance(p), rid, typ) for rid, typ, pg in polys)
            hit = [(near[0][1], near[0][2])] if near and near[0][0] < 6 else []
            status = "label_on_boundary" if hit else "missed"
        else:
            status = "ok"
        if hit:
            rid, typ = hit[0]
            others = [u for u, (ux, uy) in units.items() if u != uid and Polygon(next(r["polygon_pt"] for r in rooms if r["id"] == rid)).contains(Point(ux, uy))]
            if others:
                status = "merged"
            elif typ != exp and exp != "other":
                status = "wrong_class"
        per_unit[uid] = dict(expected=exp, room=hit[0][0] if hit else None, predicted=hit[0][1] if hit else None, status=status)
        res[exp].append(status)
    summary = {}
    for exp, st in res.items():
        n = len(st)
        c = {k: st.count(k) for k in set(st)}
        summary[exp] = dict(n_units=n, counts=c, isolated_and_correct=c.get("ok", 0) + c.get("label_on_boundary", 0))
    return dict(n_units_found=len(units), summary=summary, per_unit=per_unit)


def room_level_stats(rooms, ocr):
    """Predicted rooms that contain >=1 unit labels: how many contain exactly one (precision proxy)."""
    units = unit_labels(ocr)
    cnt = []
    for r in rooms:
        pg = Polygon(r["polygon_pt"])
        n = sum(pg.contains(Point(p)) for p in units.values())
        cnt.append(n)
    cnt = np.array(cnt)
    return dict(rooms=len(rooms), rooms_with_0_units=int((cnt == 0).sum()), rooms_with_1_unit=int((cnt == 1).sum()), rooms_with_2plus_units=int((cnt >= 2).sum()))
