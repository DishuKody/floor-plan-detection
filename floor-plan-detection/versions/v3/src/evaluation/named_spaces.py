"""Proxy check for *named* non-bedroom spaces (Stair 03, Riser, Lift 04/05, AOV, Store, Clnrs, Tutor Flat, Core 03).
The expectations below (type + plausible area range in m^2 at the printed 1:100 scale) are my manual reading of this drawing,
NOT a measured ground truth. A space passes only if its room (1) has the expected type, (2) holds no other named space,
(3) has an area inside the plausible range. Says nothing about boundary accuracy."""
import json
import re
import sys
from pathlib import Path
from shapely.geometry import Polygon, Point

# label regex -> (name, accepted types, plausible area m^2)
EXPECT = [
    (r"stair", "stair", {"stair"}, (8, 45)),
    (r"riser", "riser", {"shaft_or_riser"}, (1.5, 12)),
    (r"lift0?4", "lift04", {"lift"}, (2, 8)),
    (r"lift0?5", "lift05", {"lift"}, (2, 8)),
    (r"^aov", "aov", {"shaft_or_riser"}, (1, 8)),
    (r"store", "store", {"storage"}, (8, 45)),
    (r"clnrs", "cleaners", {"laundry_or_utility"}, (0.5, 8)),
    (r"tutorflat", "tutor_flat", {"other"}, (15, 80)),   # NOTE: ill-posed as a SINGLE room (the flat has several rooms); see tutor_flat_group
    (r"core0?3", "core03_lobby", {"corridor_or_hallway", "communal_or_shared_space"}, (20, 70)),
]


def evaluate_named(rooms, ocr):
    polys = [(r, Polygon(r["polygon_pt"])) for r in rooms]
    found = []
    for t in ocr:
        txt = re.sub(r"\s+", "", t["text"].lower())
        for rx, name, types, rng in EXPECT:
            if re.search(rx, txt) and not re.search(r"type|\d\d/d", txt):
                c = Point((t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2)
                found.append((name, types, rng, c))
                break
    names_in = {}
    out = {}
    for name, types, rng, c in found:
        room = next((r for r, p in polys if p.contains(c)), None)
        if room is None:
            room = min(polys, key=lambda rp: rp[1].distance(c))[0] if polys else None
        names_in.setdefault(room["id"], set()).add(name)
        out[name] = dict(room=room["id"], type=room["type"], area_m2=room["approx_area_m2"], expected_types=sorted(types), area_range=rng)
    res = {}
    for name, d in out.items():
        issues = []
        if d["type"] not in d["expected_types"]:
            issues.append(f"wrong_type:{d['type']}")
        if len(names_in[d["room"]]) > 1:
            issues.append("merged_with:" + "+".join(sorted(names_in[d["room"]] - {name})))
        if not (d["area_range"][0] <= d["area_m2"] <= d["area_range"][1]):
            issues.append(f"implausible_area:{d['area_m2']}m2")
        res[name] = dict(d, ok=not issues, issues=issues)
    missing = [n for _, n, *_ in EXPECT if n not in res]
    # the Tutor Flat is a small flat (living/kitchen, hall, bedroom, bathroom): judge it as the group of 'other' rooms
    grp = [r for r in rooms if r["type"] == "other"]
    tg = dict(n_rooms=len(grp), area_m2=round(sum(r["approx_area_m2"] for r in grp), 1), plausible_25_80=25 <= sum(r["approx_area_m2"] for r in grp) <= 80)
    return dict(n_ok=sum(v["ok"] for v in res.values()), n_checked=len(res), not_found_by_ocr=missing, spaces=res, tutor_flat_group=tg)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        rooms = json.load(open(Path(d) / "rooms.json"))["rooms"]
        ocr = json.load(open(Path(d) / "debug" / "ocr.json"))
        r = evaluate_named(rooms, ocr)
        json.dump(r, open(Path(d) / "named_spaces_eval.json", "w"), indent=1)
        print(d, f"{r['n_ok']}/{r['n_checked']} named spaces ok; OCR did not find: {r['not_found_by_ocr']}")
        for k, v in r["spaces"].items():
            print(f"  {k:14s} {v['room']} {v['type']:22s} {v['area_m2']:6.1f} m2  {'OK' if v['ok'] else 'FAIL ' + ','.join(v['issues'])}")
