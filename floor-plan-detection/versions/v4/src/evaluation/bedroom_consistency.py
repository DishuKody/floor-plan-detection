"""Consistency proxy: all bedrooms of one drawn unit type (e.g. B703) are the same standard layout, so their areas should be
(near-)identical after identical treatment. Large spread / leftover pockets indicate fragmented or merged bedrooms.
Also counts small non-bedroom 'pocket' rooms touching a bedroom (fragments of bedrooms)."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon


def evaluate_bedrooms(rooms):
    by = defaultdict(list)
    for r in rooms:
        if r["type"] != "bedroom":
            continue
        codes = {re.sub(r"\s+", "", c.upper()) for c in r["type_codes"]}
        for c in codes:
            by[c].append(r["approx_area_m2"])
    groups = {}
    for c, a in by.items():
        a = np.array(a)
        if len(a) >= 5:
            med = np.median(a)
            groups[c] = dict(n=len(a), median_m2=round(float(med), 2), cv=round(float(a.std() / a.mean()), 3),
                             n_outliers_gt_10pct=int((np.abs(a - med) > 0.10 * med).sum()))
    beds = [Polygon(r["polygon_pt"]) for r in rooms if r["type"] == "bedroom"]
    pockets = [r for r in rooms if r["type"] == "unknown" and r["approx_area_m2"] < 4.0
               and any(Polygon(r["polygon_pt"]).buffer(1.5).intersects(b) for b in beds)]
    return dict(groups=groups, pockets_touching_bedroom=len(pockets), pocket_ids=[p["id"] for p in pockets])


if __name__ == "__main__":
    for d in sys.argv[1:]:
        rooms = json.load(open(Path(d) / "rooms.json"))["rooms"]
        r = evaluate_bedrooms(rooms)
        json.dump(r, open(Path(d) / "bedroom_consistency.json", "w"), indent=1)
        print(d, json.dumps(r))
