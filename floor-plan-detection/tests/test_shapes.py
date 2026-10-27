import numpy as np
import pytest
from shapely.geometry import Polygon, Point
from shapely import affinity
from src.postprocessing.shapes import describe, snap_axis_aligned


def test_rectangle_square_circle_L_triangle_classification():
    assert describe(Polygon([(0, 0), (40, 0), (40, 20), (0, 20)]))["shape_type"] == "rectangle"
    assert describe(Polygon([(0, 0), (30, 0), (30, 30), (0, 30)]))["shape_type"] == "square"
    assert describe(Point(0, 0).buffer(25, 64))["shape_type"] == "circle"
    L = Polygon([(0, 0), (60, 0), (60, 20), (20, 20), (20, 60), (0, 60)])
    assert describe(L)["shape_type"] == "rectilinear_polygon"
    assert describe(Polygon([(0, 0), (50, 0), (20, 40)]))["shape_type"] == "irregular_polygon"


def test_rotated_rectangle_is_still_a_rectangle():
    r = affinity.rotate(Polygon([(0, 0), (80, 0), (80, 30), (0, 30)]), 37, origin=(0, 0))
    d = describe(r)
    assert d["shape_type"] == "rectangle" and d["rectangularity"] > 0.99 and d["axis_aligned_edge_fraction"] > 0.99


def test_descriptors_of_L_shape_are_sensible():
    L = Polygon([(0, 0), (60, 0), (60, 20), (20, 20), (20, 60), (0, 60)])
    d = describe(L)
    assert d["n_vertices"] == 6 and not d["is_convex"] and d["solidity"] < 0.8 and d["rectangularity"] < 0.7


def test_snap_makes_slightly_tilted_edges_exactly_axis_aligned_and_keeps_area():
    p = Polygon([(0, 0.0), (100, 1.2), (101, 60), (0.5, 59)])           # ~0.7 deg tilt
    q = snap_axis_aligned(p, tol_deg=3)
    xs = sorted({round(x, 6) for x, _ in q.exterior.coords})
    ys = sorted({round(y, 6) for _, y in q.exterior.coords})
    assert len(xs) == 2 and len(ys) == 2                                  # an exact axis-aligned rectangle
    assert abs(q.area - p.area) / p.area < 0.02


def test_snap_leaves_a_genuinely_diagonal_edge_and_preserves_holes():
    tri = Polygon([(0, 0), (100, 0), (100, 50), (0, 100)])               # a 26 deg slanted edge must stay slanted
    q = snap_axis_aligned(tri)
    cs = list(q.exterior.coords)
    assert any(abs(a[0] - b[0]) > 20 and abs(a[1] - b[1]) > 20 for a, b in zip(cs, cs[1:]))
    donut = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)], [[(30, 30), (31, 70), (70, 70.5), (70, 30)]])
    qd = snap_axis_aligned(donut)
    assert len(qd.interiors) == 1 and qd.is_valid
