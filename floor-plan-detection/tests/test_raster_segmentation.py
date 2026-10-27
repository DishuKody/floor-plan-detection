"""Boundary accuracy of the raster segmentation against a synthetic plan with EXACTLY known room rectangles."""
import cv2
import numpy as np
import pytest
from shapely.geometry import box, Polygon
from src.raster.segmentation import PageTransform, close_wall_gaps, thick_walls, segment_rooms, mask_to_polygon
from src.postprocessing.shapes import snap_axis_aligned
from tests.conftest import SYN_ROOMS, SYN_LABELS, SYN_W, SYN_H, synthetic_plan


def _text_boxes():
    out = {}
    for name, (txt, (x, y)) in SYN_LABELS.items():
        (w, h), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.1, 3)
        out[name] = (x, y - h, x + w, y + base)
    return out


def _seeds():
    return [dict(cls=n, text=SYN_LABELS[n][0], xy=((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)) for n, b in _text_boxes().items()]


def _hausdorff_px(a: Polygon, b: Polygon):
    return max(a.exterior.hausdorff_distance(b.exterior), b.exterior.hausdorff_distance(a.exterior))


@pytest.fixture(scope="module")
def seg():
    g = synthetic_plan()
    return segment_rooms(g, _seeds(), list(_text_boxes().values()), {})


def test_every_room_matches_its_ground_truth_rectangle(seg):
    assert len(seg["rooms"]) == 3
    names = list(SYN_ROOMS)
    for r in seg["rooms"]:
        gt = box(*SYN_ROOMS[names[r["seed_ids"][0]]])
        poly = snap_axis_aligned(r["poly"])
        iou = poly.intersection(gt).area / poly.union(gt).area
        assert iou >= 0.985, (names[r["seed_ids"][0]], iou)
        assert _hausdorff_px(poly, gt) <= 2.5                                # boundary within 2.5 px of the true outline everywhere
        assert len(poly.interiors) == 0                                      # bold text did not punch holes
        # a rectangle stays a rectangle plus at most one shallow threshold recess per doorway (half the 12 px wall slab): no fragmented outline
        assert len(poly.exterior.coords) - 1 <= 12
        assert poly.symmetric_difference(gt).area <= 2 * 60 * 6 + 0.01 * gt.area


def test_door_openings_are_closed_by_straight_lines_not_a_curved_partition(seg):
    kinds = {c["kind"] for c in seg["closures"]}
    assert "door_threshold" in kinds
    assert all(r["kind"] == "single" for r in seg["rooms"])                   # no geodesic (label-proximity) split was needed
    cl = [c for c in seg["closures"] if c["gap_px"] > 40]
    assert len(cl) >= 2
    assert all(abs(c["p0"][0] - c["p1"][0]) < 1.5 or abs(c["p0"][1] - c["p1"][1]) < 1.5 for c in cl)     # axis-aligned


def test_closure_crosses_the_whole_opening_regression_for_start_inside_wall_bug():
    """Earlier bug: rays started inside the wall end (the skeleton lies half a thickness inside) and logged gap 0.
    The closure must span the 60 px opening."""
    g = synthetic_plan(with_text=False)
    walls = thick_walls(g)["walls"]
    closed, lines = close_wall_gaps(walls, 6, SYN_W)
    big = [l for l in lines if l["gap_px"] > 40]
    assert len(big) >= 2 and all(40 < l["gap_px"] < 70 for l in big)
    assert closed[130, 400] == 1 and closed[300, 530] == 1                    # both door openings are sealed


def test_parallel_walls_are_never_bridged():
    g = np.full((300, 600), 255, np.uint8)
    cv2.rectangle(g, (50, 100), (550, 112), 0, -1)                           # two parallel walls 60 px apart, no ends facing each other
    cv2.rectangle(g, (50, 170), (550, 182), 0, -1)
    walls = thick_walls(g)["walls"]
    closed, lines = close_wall_gaps(walls, 6, 600)
    assert closed[140, 300] == 0 and closed[140, 100] == 0                   # the corridor between them stays open


def test_text_is_not_wall_and_a_wall_under_a_label_is_kept():
    g = synthetic_plan()
    tw = thick_walls(g, list(_text_boxes().values()))
    for n, b in _text_boxes().items():
        x0, y0, x1, y1 = [int(v) for v in b]
        assert tw["walls"][y0 + 2:y1 - 2, x0 + 2:x1 - 2].sum() == 0, f"text '{n}' leaked into the wall mask"
    assert tw["walls"][400, 400] == 1 and tw["walls"][50, 300] == 1          # real walls survive


def test_title_text_and_logo_box_far_from_the_plan_do_not_become_rooms():
    g = synthetic_plan()
    big = np.full((SYN_H + 200, SYN_W), 255, np.uint8)
    big[:SYN_H] = g
    title = "TYPICAL FLOOR PLAN"
    cv2.putText(big, title, (60, SYN_H + 60), cv2.FONT_HERSHEY_DUPLEX, 1.4, 0, 4, cv2.LINE_AA)
    (tw_, th_), tb_ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.4, 4)
    title_box = (60, SYN_H + 60 - th_, 60 + tw_, SYN_H + 60 + tb_)             # the real pipeline passes every OCR box, titles included
    cv2.rectangle(big, (480, SYN_H + 20), (780, SYN_H + 110), 0, 3)           # logo-like frame, thin lines, not attached to any wall
    s = segment_rooms(big, _seeds(), list(_text_boxes().values()) + [title_box], {})
    assert all(r["kind"] != "unlabelled" for r in s["rooms"])
    assert s["footprint"][SYN_H + 40:, :].sum() == 0                          # the footprint stops at the plan


def test_mask_to_polygon_is_pixel_exact_and_keeps_real_holes():
    m = np.zeros((100, 100), np.uint8)
    m[20:80, 10:90] = 1
    m[40:60, 30:50] = 0                                                       # a real 20x20 hole
    p = mask_to_polygon(m, simplify=0.5, min_hole_area=50)
    assert p.bounds == pytest.approx((9.5, 19.5, 89.5, 79.5), abs=0.01)       # pixel centres pushed out by 0.5 px
    assert len(p.interiors) == 1 and abs(Polygon(p.interiors[0]).area - 20 * 20) < 25
    assert abs(p.area - (60 * 80 - 400)) < 20


def test_page_transform_roundtrip_and_known_points():
    tf = PageTransform((50.0, 100.0, 450.0, 400.0), 800, 600, 2.0)           # image 800x600 placed at 400x300 pt
    assert tf.sx == pytest.approx(0.5) and tf.sy == pytest.approx(0.5)
    assert tf.img_to_pt([0, 0]).tolist() == [50.0, 100.0] and tf.img_to_pt([800, 600]).tolist() == [450.0, 400.0]
    assert tf.img_to_page_px([400, 300]).tolist() == [500.0, 500.0]           # (250 pt, 250 pt) * 2
    pts = np.random.default_rng(0).uniform(0, 600, (20, 2))
    assert np.allclose(tf.page_px_to_img(tf.img_to_page_px(pts)), pts) and np.allclose(tf.pt_to_img(tf.img_to_pt(pts)), pts)
