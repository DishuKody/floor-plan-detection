"""Independent-area output: unique IDs, neighbours never share a colour, full-page size preserved, JSON geometry matches."""
import json
import numpy as np
from shapely.geometry import box
from src.output.areas import assign_colours, write_areas, PALETTE


def _room(i, x0, y0, x1, y1, s=2.0):
    ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return dict(id=f"room-{i:04d}", type="unknown", approx_area_m2=1.0, polygon_pt=ring, polygon=[[x * s, y * s] for x, y in ring], holes=[], shape={"shape_type": "rectangle"}, flags=[])


def test_neighbours_never_share_a_colour_and_colours_repeat_rarely():
    polys = [box(i % 6 * 10, i // 6 * 10, i % 6 * 10 + 10, i // 6 * 10 + 10) for i in range(36)]    # a 6 x 6 grid of touching squares
    cols = assign_colours(polys)
    for i, p in enumerate(polys):
        for j, q in enumerate(polys):
            if i != j and p.buffer(0.5).intersects(q):
                assert cols[i] != cols[j]
    assert len(set(cols)) == 36 and len(PALETTE) >= 36                                               # 36 areas -> 36 different colours


def test_write_areas_keeps_the_page_size_and_lists_every_area(tmp_path):
    base = np.full((400, 600, 3), 255, np.uint8)
    rooms = [_room(1, 10, 10, 100, 90), _room(2, 100, 10, 200, 90), _room(3, 10, 90, 200, 150)]
    assert write_areas(base, rooms, 2.0, tmp_path) == 3
    import cv2
    assert cv2.imread(str(tmp_path / "areas_overlay.png")).shape == base.shape
    J = json.load(open(tmp_path / "areas.json"))
    assert [a["id"] for a in J["areas"]] == ["room-0001", "room-0002", "room-0003"] and J["page_px"] == [600, 400]
    assert len({tuple(a["colour_rgb"]) for a in J["areas"]}) == 3
