"""The primary overlay / mask must be the ENTIRE page at exactly the render size; the detection ROI only restricts detection."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest

OUT = Path(__file__).resolve().parent.parent / "outputs"
pytestmark = pytest.mark.skipif(not (OUT / "overlay.png").exists(), reason="run the pipeline first")


def _load():
    J = json.load(open(OUT / "rooms.json"))
    return J, cv2.imread(str(OUT / "overlay.png")), cv2.imread(str(OUT / "debug" / "render_reference.png")), cv2.imread(str(OUT / "room_masks.png"))


def test_outputs_have_exactly_the_page_render_size_and_aspect():
    J, ov, ref, mk = _load()
    img = J["image"]
    assert ov.shape == ref.shape == mk.shape
    assert (ov.shape[1], ov.shape[0]) == (img["width"], img["height"])
    pw, ph = img["transform"]["page_size_pt"]
    s = img["transform"]["scale_px_per_pt"]
    assert abs(ov.shape[1] - pw * s) <= 1 and abs(ov.shape[0] - ph * s) <= 1          # same aspect as the PDF page, no stretching
    assert img["output_canvas"] == "full_page"
    x0, y0, x1, y1 = img["detection_roi_pt"]
    assert (x1 - x0) * s < ov.shape[1] and (y1 - y0) * s < ov.shape[0]                  # ROI is strictly smaller than the canvas


def test_overlay_is_the_page_render_plus_paint_only():
    """Outside painted rooms / elements / legend the overlay must be pixel-identical to the plain render (no crop, shift or resample)."""
    J, ov, ref, _ = _load()
    E = json.load(open(OUT / "elements.json"))["elements"]
    s = J["image"]["transform"]["scale_px_per_pt"]
    painted = np.zeros(ov.shape[:2], np.uint8)
    for r in J["rooms"]:
        for p in [r["polygon"]] + r.get("extra_polygons", []):
            cv2.fillPoly(painted, [np.array(p).astype(np.int32)], 255)
    painted = cv2.dilate(painted, np.ones((9, 9), np.uint8))
    for e in E:
        if e["type"] == "door":
            cv2.circle(painted, tuple((np.array(e["hinge_pt"]) * s).astype(int)), 8, 255, -1)
        elif e["type"] == "window":
            cv2.fillPoly(painted, [(np.array(e["polygon_pt"]) * s).astype(np.int32)], 255)
            painted = cv2.dilate(painted, np.ones((5, 5), np.uint8))
        elif e["type"] == "column":
            b = (np.array(e["bbox_pt"]) * s).astype(int)
            cv2.rectangle(painted, (b[0] - 4, b[1] - 4), (b[2] + 4, b[3] + 4), 255, -1)
    x0, y0, x1, y1 = J["image"]["legend"]["box_px"]
    painted[y0 - 2:y1 + 3, x0 - 2:x1 + 3] = 255
    diff = (np.abs(ov.astype(int) - ref.astype(int)).sum(2) > 0) & (painted == 0)
    assert int(diff.sum()) == 0
    assert (ov[:3] == ref[:3]).all() and (ov[-3:] == ref[-3:]).all() and (ov[:, :3] == ref[:, :3]).all() and (ov[:, -3:] == ref[:, -3:]).all()


def test_legend_does_not_cover_drawing():
    J, *_ = _load()
    assert J["image"]["legend"]["ink_fraction_under_legend"] < 0.01


def test_room_mask_matches_polygons_in_page_coordinates():
    from shapely.geometry import Polygon
    J, _, _, mk = _load()
    sc = J["image"]["transform"]["scale_px_per_pt"]
    for r in J["rooms"]:
        pg = Polygon(r["polygon_pt"], r["holes"] and None)
        p = np.array(pg.representative_point().coords[0]) * sc               # a point guaranteed inside the room, in pixels
        assert mk[int(p[1]), int(p[0])].tolist() != [255, 255, 255], r["id"]
