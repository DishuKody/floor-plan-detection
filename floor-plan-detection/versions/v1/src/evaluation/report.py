"""Usage: python -m src.evaluation.report <outputs_dir> [<outputs_dir_ablation>]
Computes proxy (unit-ID label) metrics + geometry sanity checks and writes <outputs_dir>/evaluation.json."""
import json
import sys
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from .unit_id_proxy import evaluate, room_level_stats


def geometry_checks(rooms):
    polys = [Polygon(r["polygon_pt"]) for r in rooms]
    invalid = sum(not p.is_valid for p in polys)
    tree = STRtree(polys)
    overlaps = 0
    for i, p in enumerate(polys):
        for j in tree.query(p):
            if j > i and p.intersection(polys[j]).area > 0.02 * min(p.area, polys[j].area):
                overlaps += 1
    return dict(n_rooms=len(polys), invalid_polygons=int(invalid), pairs_overlapping_gt_2pct=overlaps,
                rooms_with_holes=sum(1 for r in rooms if r["holes"]))


def run(out_dir):
    out = Path(out_dir)
    rooms = json.load(open(out / "rooms.json"))["rooms"]
    ocr = json.load(open(out / "debug" / "ocr.json"))
    ev = evaluate(rooms, ocr)
    res = dict(proxy_unit_id=ev["summary"], n_unit_labels_read=ev["n_units_found"], room_level=room_level_stats(rooms, ocr),
               geometry=geometry_checks(rooms),
               by_type={t: sum(r["type"] == t for r in rooms) for t in sorted({r["type"] for r in rooms})},
               review_required=sum(r["review_required"] for r in rooms),
               per_unit_failures={u: v for u, v in ev["per_unit"].items() if v["status"] != "ok"})
    json.dump(res, open(out / "evaluation.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(d); print(json.dumps(run(d), indent=1))
