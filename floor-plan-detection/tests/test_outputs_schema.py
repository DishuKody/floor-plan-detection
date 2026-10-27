"""Schema / validity checks on a generated outputs/ directory (run the pipeline first)."""
import json
from pathlib import Path as P
import pytest
from shapely.geometry import Polygon
from shapely.strtree import STRtree

OUT = P(__file__).resolve().parent.parent / "outputs"
pytestmark = pytest.mark.skipif(not (OUT / "rooms.json").exists(), reason="run the pipeline first")

ROOM_KEYS = {"id", "type", "polygon", "polygon_pt", "holes", "confidence", "confidence_source", "area_pixels", "review_required", "source"}


def test_rooms_json_schema_and_coordinates():
    d = json.load(open(OUT / "rooms.json"))
    img = d["image"]
    assert img["coordinate_origin"] == "top_left" and img["units"] == "pixels"
    s = img["transform"]["scale_px_per_pt"]
    ids = set()
    for r in d["rooms"]:
        assert ROOM_KEYS <= set(r), ROOM_KEYS - set(r)
        assert r["id"] not in ids; ids.add(r["id"])
        assert r["confidence"] is None                     # no calibrated probability is claimed
        for (x, y), (xp, yp) in zip(r["polygon"], r["polygon_pt"]):
            assert 0 <= x <= img["width"] and 0 <= y <= img["height"]
            assert x == pytest.approx(xp * s, abs=0.2) and y == pytest.approx(yp * s, abs=0.2)


def test_room_polygons_valid_and_non_overlapping():
    rooms = json.load(open(OUT / "rooms.json"))["rooms"]
    polys = [Polygon(r["polygon_pt"], r["holes"] and []) for r in rooms]
    assert all(p.is_valid and p.area > 0 for p in polys)
    tree = STRtree(polys)
    for i, p in enumerate(polys):
        for j in tree.query(p):
            if j > i:
                assert p.intersection(polys[j]).area <= 0.02 * min(p.area, polys[j].area)


def test_elements_json_has_typed_records():
    e = json.load(open(OUT / "elements.json"))["elements"]
    assert {"door", "column"} <= {x["type"] for x in e}
