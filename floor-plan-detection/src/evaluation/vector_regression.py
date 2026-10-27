"""Usage: python -m src.evaluation.vector_regression <old_outputs_dir> <new_outputs_dir>
Room-by-room comparison of two vector-mode runs (matched by id): type equality, polygon IoU, max vertex displacement (pt), element counts."""
import json
import sys
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon


def compare(old, new):
    A = json.load(open(Path(old) / "rooms.json"))["rooms"]
    B = {r["id"]: r for r in json.load(open(Path(new) / "rooms.json"))["rooms"]}
    ious, same_type, disp, missing = [], 0, [], []
    for a in A:
        b = B.get(a["id"])
        if b is None:
            missing.append(a["id"])
            continue
        pa, pb = Polygon(a["polygon_pt"]), Polygon(b["polygon_pt"])
        u = pa.union(pb).area
        ious.append(pa.intersection(pb).area / u if u else 1.0)
        same_type += a["type"] == b["type"]
        if len(a["polygon_pt"]) == len(b["polygon_pt"]):
            disp.append(float(np.abs(np.array(a["polygon_pt"]) - np.array(b["polygon_pt"])).max()))
    ea = json.load(open(Path(old) / "elements.json"))["elements"]
    eb = json.load(open(Path(new) / "elements.json"))["elements"]
    cnt = lambda E: {k: sum(e["type"] == k for e in E) for k in sorted({e["type"] for e in E})}
    return dict(rooms_old=len(A), rooms_new=len(B), missing_in_new=len(missing), only_in_new=len(set(B) - {a["id"] for a in A}),
                same_type=f"{same_type}/{len(A) - len(missing)}", min_iou=round(float(min(ious)), 4), mean_iou=round(float(np.mean(ious)), 4),
                max_vertex_displacement_pt=(round(max(disp), 3) if disp else None), elements_old=cnt(ea), elements_new=cnt(eb))


if __name__ == "__main__":
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), indent=1))
