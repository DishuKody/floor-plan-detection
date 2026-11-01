"""v6: dash-chain filter, long thin lines (glazed screens), robust region polygons, annotated-reference evaluation, end-to-end on a synthetic plan."""
from pathlib import Path
import cv2
import numpy as np
import pymupdf
import pytest
import yaml
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box

from src.postprocessing.polygons import largest_polygon, region_polygon
from src.vector.extract_pdf_paths import Path as VPath
from src.vector.walls import find_dash_chains, find_long_thin_lines
from src.evaluation import expected_ref as E
from tests.test_vector_v5 import synthetic_vector_pdf

ROOT = Path(__file__).resolve().parent.parent
CLASSES = yaml.safe_load(open(ROOT / "config" / "classes.yaml"))["classes"]
PT_PER_M = 14.17                                  # 1:200


def _open_seg(page, a, b, w):
    sh = page.new_shape()
    sh.draw_line(a, b)
    sh.finish(width=w, color=(0, 0, 0), closePath=False)                          # a genuine open segment (the default closes it as a -> b -> a)
    sh.commit()


def seg(a, b, w=0.4):
    return VPath(np.array([a, b], float), False, w, None, None)


def run_line(start, direction, lengths, gaps, w=0.4):
    out, p = [], np.array(start, float)
    d = np.array(direction, float) / np.hypot(*direction)
    for i, L in enumerate(lengths):
        out.append(seg(p, p + d * L, w))
        p = p + d * (L + (gaps[i] if i < len(gaps) else 0))
    return out


def test_dash_chain_matches_regular_dash_dot_at_any_angle():
    P = run_line((10, 10), (3, 2), [12, 2] * 6, [4, 4] * 6)                     # dash - dot, equal gaps, 34 degrees
    assert len(find_dash_chains(P, 0.3, PT_PER_M)) == len(P)
    P = run_line((10, 10), (1, 0), [20] * 8, [5] * 8)                           # plain dashes
    assert len(find_dash_chains(P, 0.3, PT_PER_M)) == len(P)


def test_dash_chain_leaves_walls_and_irregular_openings_alone():
    wall = run_line((0, 0), (1, 0), [30] * 10, [0] * 10)                        # a wall drawn in pieces: touching ends
    assert find_dash_chains(wall, 0.3, PT_PER_M) == set()
    openings = run_line((0, 0), (1, 0), [40, 25, 60, 30, 45, 35, 50], [12, 45, 100, 25, 70, 8, 30])   # window / door openings: irregular gaps
    assert find_dash_chains(openings, 0.3, PT_PER_M) == set()
    assert find_dash_chains(run_line((0, 0), (1, 0), [10] * 4, [5] * 4), 0.3, PT_PER_M) == set()          # too short a chain


def test_long_thin_lines_inside_region_only_and_not_in_series():
    region = box(0, 0, 500, 500)
    thin = 0.1
    P = [seg((50, 50), (50, 250), thin),                                        # 14 m screen            -> structure
         seg((100, 100), (110, 100), thin),                                     # furniture edge          -> no
         seg((600, 50), (600, 250), thin),                                      # outside the building    -> no
         seg((200, 50), (200, 250), 1.0)]                                       # structural weight       -> not this rule's job
    P += [seg((300 + 8 * i, 50), (300 + 8 * i, 250), thin) for i in range(6)]    # hatch: 6 parallel lines 8 pt apart -> series, dropped
    got = find_long_thin_lines(P, set(), 0.08, 0.15, PT_PER_M, region, min_len_m=4.0)
    assert got == {0}


def test_largest_polygon_flattens_nested_collections():
    nested = GeometryCollection([MultiPolygon([box(0, 0, 10, 10), box(20, 0, 22, 2)]), Polygon([(0, 0), (5, 0)] + [(5, 0)]).boundary])
    assert largest_polygon(nested).area == 100
    assert largest_polygon(GeometryCollection()) is None


def test_region_polygon_survives_pinched_contours():
    """Two blobs joined by a single diagonal pixel: the contour touches itself; the region must not be lost (it used to be dropped)."""
    lab = np.zeros((40, 40), np.int32)
    lab[5:20, 5:20] = 1                                                        # large blob
    lab[20:26, 20:26] = 1                                                      # small blob touching the large one at one corner pixel
    poly = region_polygon(lab, 1, (5, 5, 21, 21), (0.0, 0.0), 1.0)
    assert poly is not None and poly.area >= 14 * 14


def test_expected_ref_networks_mode_separates_cross_from_line_network():
    img = np.full((300, 400, 3), 255, np.uint8)
    red = (0, 0, 255)
    cv2.line(img, (40, 40), (180, 180), red, 6)                                # a hand-drawn cross: 4 ends, one crossing
    cv2.line(img, (180, 40), (40, 180), red, 6)
    cv2.line(img, (220, 60), (380, 60), red, 6)                                # a network: a T of three strokes
    cv2.line(img, (300, 60), (300, 200), red, 6)
    cv2.circle(img, (30, 250), 4, red, -1)                                     # a review dot: ignored
    marks = E.extract_marks(img, networks=True)
    assert len(marks["x_marks"]) == 1 and len(marks["lines"]) == 1


def test_separation_metric_distinguishes_a_split_from_a_leak():
    shape = (200, 400)
    A = np.array([[1, 0, 0], [0, 1, 0]], float)                                # identity registration
    split = [dict(polygon=[[10, 10], [195, 10], [195, 190], [10, 190]]), dict(polygon=[[205, 10], [390, 10], [390, 190], [205, 190]])]
    leak = [dict(polygon=[[10, 10], [390, 10], [390, 190], [10, 190]])]
    line = dict(pts=np.array([[200.0, y] for y in range(30, 170)]), bbox=[199, 30, 201, 170])
    assert E.evaluate_separation(dict(lines=[line]), A, split, shape, offset_px=8)["separated_both_in_rooms"] == 1.0
    assert E.evaluate_separation(dict(lines=[line]), A, leak, shape, offset_px=8)["separated"] == 0.0


def _run(pdf, out, long_thin):
    from src.pipeline import run_vector
    cfg = yaml.safe_load(open(ROOT / "config" / "inference.yaml"))
    cfg["compute"] = {"device": "cpu", "sample_gpu": False}
    cfg["ocr"]["enabled"] = False
    cfg["drawing"]["scale_ratio"] = 200
    cfg["vector"]["long_thin_lines"]["enabled"] = long_thin
    return run_vector(str(pdf), cfg, CLASSES, out, log=lambda *a: None)[0]


def test_glazed_screen_in_the_clutter_weight_splits_a_room_end_to_end(tmp_path):
    pdf, pm, (w, h) = synthetic_vector_pdf(tmp_path / "s.pdf", 200)
    doc = pymupdf.open(str(pdf))
    page = doc[0]
    x_mid = 60.0 + w / 2                                                        # a screen across the first room, from wall to wall, in the 0.1 pt clutter weight
    _open_seg(page, (x_mid, 250.0), (x_mid, 250.0 + h), 0.1)
    doc.saveIncr()
    n_off = len([r for r in _run(pdf, tmp_path / "off", False) if r["approx_area_m2"] > 20])
    n_on = len([r for r in _run(pdf, tmp_path / "on", True) if r["approx_area_m2"] > 15])
    assert n_off == 3 and n_on == 4
