import numpy as np
import cv2
import pytest
from shapely.geometry import Polygon
from src.vector.doors import detect_door_arcs, fit_circle
from src.vector.extract_pdf_paths import Path
from src.vector.walls import find_grid_paths, extend_dangling_walls
from src.postprocessing.polygons import region_polygon, polygon_to_record
from src.postprocessing.instances import split_region


def _path(pts, width=0.48):
    return Path(np.array(pts, float), False, width, (0, 0, 0), None)


def test_region_polygon_maps_back_to_page_points():
    roi, S = (100, 200, 300, 300), 4
    lab = np.zeros((400, 800), np.int32)
    lab[40:120, 80:240] = 1                      # px box -> pt: x 120..160, y 210..230
    poly = region_polygon(lab, 1, (80, 40, 160, 80), roi, S)
    x0, y0, x1, y1 = poly.bounds
    assert (x0, y0, x1, y1) == pytest.approx((120, 210, 160, 230), abs=0.6)
    rec = polygon_to_record(poly, 1.5)
    assert np.array(rec["polygon"]).max(0) == pytest.approx([160 * 1.5, 230 * 1.5], abs=1.0)   # pixel = pt * render scale
    assert Polygon(rec["polygon_pt"]).is_valid


def test_door_arc_detected_from_segmented_quarter_circle():
    c, r = np.array([50.0, 60.0]), 26.0
    a = np.radians(np.linspace(0, 90, 12))
    arc = np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)]
    arcs = detect_door_arcs([_path(arc)])
    assert len(arcs) == 1 and arcs[0].radius == pytest.approx(r, abs=0.5) and arcs[0].sweep_deg == pytest.approx(90, abs=3)
    assert detect_door_arcs([_path(arc[:5])]) == []          # too few points -> not an arc
    cc, rr, res = fit_circle(arc)
    assert np.allclose(cc, c, atol=0.1) and res < 1e-6


def test_grid_dash_chain_removed_but_short_wall_kept():
    dashes = [_path([[10, y], [10, y + 20]], 0.24) for y in range(0, 200, 26)]
    wall = [_path([[40, 0], [40, 30]], 0.24), _path([[40, 36], [40, 60]], 0.24)]
    g = find_grid_paths(dashes + wall)
    assert g == set(range(len(dashes)))


def test_dangling_wall_extends_to_collinear_wall_gap():
    roi, S = (0, 0, 100, 100), 4
    mask = np.zeros((400, 400), np.uint8)
    cv2.line(mask, (0, 280), (150, 280), 255, 1)       # horizontal wall y=70pt, with a gap at x 150..250 px
    cv2.line(mask, (250, 280), (400, 280), 255, 1)
    stem = _path([[50, 5], [50, 60]], 0.72)             # vertical stem ending 10pt short, inside the gap
    stem2 = _path([[50, 5], [50, 60]], 0.72)
    added = extend_dangling_walls([stem], set(), mask, roi, S)
    assert len(added) == 1 and added[0][1][1] == pytest.approx(70, abs=1.5)


def test_watershed_split_cuts_between_two_seeds():
    roi, S = (0, 0, 100, 50), 4
    lab = np.zeros((200, 400), np.int32)
    lab[20:180, 20:380] = 1
    lab[20:100, 190:210] = 0                            # notch: neck only in the lower half
    parts = split_region(lab, 1, (20, 20, 360, 160), [(20, 25), (80, 25)], roi, S)
    assert len(parts) == 2 and all(p is not None and p.is_valid for p in parts)
    assert parts[0].intersection(parts[1]).area < 1.0 and parts[0].bounds[2] <= parts[1].bounds[2] + 1


# ---- v2 regression tests -------------------------------------------------------------------------------------------
from src.vector.walls import rasterize_mask
from src.vector.candidate_regions import close_doors
from src.vector.doors import DoorArc


def test_fragmented_door_arc_is_merged_into_one_door():
    """A swing stored as two path fragments (37/39 deg each in the real riser door) must still be found as one ~90 deg door."""
    c, r = np.array([50.0, 60.0]), 26.0
    frag = lambda a0, a1: _path(np.c_[c[0] + r * np.cos(np.radians(np.linspace(a0, a1, 6))), c[1] + r * np.sin(np.radians(np.linspace(a0, a1, 6)))])
    arcs = detect_door_arcs([frag(0, 40), frag(50, 90)])
    assert len(arcs) == 1 and arcs[0].n_fragments == 2 and 80 <= arcs[0].sweep_deg <= 92
    assert detect_door_arcs([frag(0, 40)]) == []              # a single 40 deg fragment is not a door


def test_text_glyphs_are_not_rasterised_as_walls_but_poche_is():
    roi, S = (0, 0, 100, 100), 4
    ang = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    glyph = Path(np.c_[50 + 7 * np.cos(ang), 50 + 5 * np.sin(ang)], True, 0.0, None, (0.0, 0.0, 0.0))     # 14-point 'heading letter'
    poche = Path(np.array([[10, 10], [40, 10], [40, 14], [10, 14]], float), True, 0.0, None, (0.0, 0.0, 0.0))
    assert rasterize_mask([glyph], set(), roi, S).sum() == 0
    assert rasterize_mask([poche], set(), roi, S).sum() > 0


def test_door_chord_follows_the_wall_not_the_open_leaf_and_reaches_far_jamb():
    roi, S = (0, 0, 100, 100), 4
    mask = np.zeros((400, 400), np.uint8)
    cv2.line(mask, (80, 0), (80, 120), 255, 1)          # wall x=20pt above the hinge (y 0..30)
    cv2.line(mask, (80, 236), (80, 400), 255, 1)        # far jamb starts y=59pt: opening is y 30..59 (29pt) but leaf radius is only 26
    arc = DoorArc(center=np.array([20.0, 30.0]), radius=26.0, p_start=np.array([46.0, 30.0]), p_end=np.array([20.0, 56.0]), sweep_deg=90.0, path_index=0, width=0.48)
    (c, e, hits, a), = close_doors(mask, [arc], roi, S)
    assert e[0] == pytest.approx(20.0, abs=0.5)          # along the wall (x constant), not the horizontal open leaf
    assert e[1] >= 57.5                                  # extended past the swing radius to the far jamb
