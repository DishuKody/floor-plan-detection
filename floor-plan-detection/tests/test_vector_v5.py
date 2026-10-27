"""v5: fast path extraction, sheet calibration, scale-aware parameters, direction-agnostic gap closing, circles, profiles."""
from pathlib import Path
import cv2
import numpy as np
import pymupdf
import pytest
import yaml
from shapely.geometry import Polygon

from src.vector.extract_pdf_paths import Path as VPath, extract_paths, extract_paths_reference
from src.vector.calibrate import structural_stroke, plan_footprint
from src.vector.elements import circle_elements
from src.postprocessing.classify import classify_room, match_keywords
from src.raster.segmentation import close_wall_gaps
from src.cli import deep_merge

ROOT = Path(__file__).resolve().parent.parent
CLASSES = yaml.safe_load(open(ROOT / "config" / "classes.yaml"))["classes"]


def _seg(page, a, b, w, color=(0, 0, 0)):
    sh = page.new_shape()
    sh.draw_line(a, b)
    sh.finish(width=w, color=color)
    sh.commit()


def synthetic_vector_pdf(path, ratio=200):
    """A single-storey strip of 3 rooms (8 m x 6 m at the printed scale, walls drawn as many short 1 pt strokes), 0.1 pt hatching as
    clutter, one long site line and an isolated decorative loop - page 842 x 595 pt."""
    doc = pymupdf.open()
    page = doc.new_page(width=842, height=595)
    pm = 28.346 * 100 / ratio                       # pt per metre
    w, h = 8 * pm, 6 * pm
    x0, y0 = 60.0, 250.0
    xs = [x0 + i * w for i in range(4)]
    step = 5.0
    for x in xs:                                    # vertical walls
        for y in np.arange(y0, y0 + h, step):
            _seg(page, (x, y), (x, min(y + step, y0 + h)), 1.0)
    for y in (y0, y0 + h):                          # horizontal walls
        for x in np.arange(xs[0], xs[-1], step):
            _seg(page, (x, y), (min(x + step, xs[-1]), y), 1.0)
    for i in range(1500):                           # clutter: thin hatching inside the first room
        yy = y0 + 3 + (i % 100) * (h - 6) / 100
        _seg(page, (xs[0] + 3 + (i // 100) * 6, yy), (xs[0] + 8 + (i // 100) * 6, yy + 1), 0.1)
    _seg(page, (10, 40), (830, 60), 1.0)            # long site line
    sh = page.new_shape()                           # isolated loop, no structure inside
    sh.draw_circle((650, 120), 60)
    sh.finish(width=1.0, color=(0, 0, 0))
    sh.commit()
    doc.save(str(path))
    return path, pm, (w, h)


def test_fast_extraction_equals_reference(tmp_path):
    pdf, *_ = synthetic_vector_pdf(tmp_path / "s.pdf")
    doc = pymupdf.open(str(pdf))
    page = doc[0]
    sh = page.new_shape()
    sh.draw_bezier((50, 500), (80, 450), (120, 450), (150, 500))
    sh.draw_rect(pymupdf.Rect(300, 500, 360, 540))
    sh.draw_quad(pymupdf.Quad(pymupdf.Point(400, 500), pymupdf.Point(450, 505), pymupdf.Point(400, 540), pymupdf.Point(450, 545)))
    sh.finish(width=0.5, color=(0, 0, 0), fill=(0.6, 0.6, 0.6))
    sh.commit()
    A, B = extract_paths_reference(page), extract_paths(page)
    assert len(A) == len(B) > 100
    for a, b in zip(A, B):
        assert a.pts.shape == b.pts.shape
        assert np.abs(a.pts - b.pts).max() < 1e-4
        assert abs(a.width - b.width) < 1e-9 and bool(a.closed) == bool(b.closed) and a.has_curve == b.has_curve


def test_fast_extraction_applies_page_rotation(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    _seg(page, (10, 20), (110, 20), 1.0)
    page.set_rotation(90)
    A, B = extract_paths_reference(page), extract_paths(page)
    assert np.abs(A[0].pts - B[0].pts).max() < 1e-4
    assert B[0].pts[:, 0].max() <= 200 + 1e-6      # displayed (rotated) coordinates


def test_structural_stroke_is_relative_to_dominant_weight():
    def paths(dom):
        out = [VPath(np.array([[0, i], [100, i]], float), False, dom, None, None) for i in range(20)]
        out += [VPath(np.array([[0, 100 + i], [30, 100 + i]], float), False, dom * 3, None, None) for i in range(5)]
        return out
    for dom in (0.12, 0.21, 0.35):
        thr, d, _ = structural_stroke(paths(dom))
        assert d == pytest.approx(dom, abs=0.006) and dom < thr < 3 * dom


def test_footprint_keeps_building_and_drops_site_line_and_empty_loop(tmp_path):
    pdf, pm, (w, h) = synthetic_vector_pdf(tmp_path / "s.pdf")
    page = pymupdf.open(str(pdf))[0]
    P = extract_paths(page)
    thr, dom, _ = structural_stroke(P)
    assert dom == pytest.approx(0.1, abs=0.006)
    fp, info = plan_footprint(P, page.rect, round(3.3 * dom, 3), 200)
    assert fp is not None
    b = Polygon([(60, 250), (60 + 3 * w, 250), (60 + 3 * w, 250 + h), (60, 250 + h)])
    assert fp.intersection(b).area / b.area > 0.9                    # the building is in
    assert fp.intersection(Polygon([(590, 60), (710, 60), (710, 180), (590, 180)])).area == 0   # the empty loop is out
    assert fp.bounds[1] > 200                                      # the long site line (y 40..60) is not part of it


def test_pipeline_runs_at_1_to_200_and_areas_scale_with_the_printed_scale(tmp_path):
    from src.pipeline import run_vector
    res = {}
    for ratio in (100, 200):
        pdf, pm, (w, h) = synthetic_vector_pdf(tmp_path / f"s{ratio}.pdf", ratio)
        cfg = yaml.safe_load(open(ROOT / "config" / "inference.yaml"))
        cfg["compute"] = {"device": "cpu", "sample_gpu": False}
        cfg["ocr"]["enabled"] = False
        cfg["drawing"]["scale_ratio"] = ratio
        rooms, elements, stats = run_vector(str(pdf), cfg, CLASSES, tmp_path / f"out{ratio}", log=lambda *a: None)
        res[ratio] = rooms
        big = [r for r in rooms if r["approx_area_m2"] > 20]
        assert len(big) == 3, [r["approx_area_m2"] for r in rooms]
        for r in big:                                              # 8 m x 6 m rooms minus half a wall each side
            assert 38 <= r["approx_area_m2"] <= 50
    # the same real-world rooms are found at both printed scales
    assert sorted(round(r["approx_area_m2"]) for r in res[100] if r["approx_area_m2"] > 20) == pytest.approx(
        sorted(round(r["approx_area_m2"]) for r in res[200] if r["approx_area_m2"] > 20), abs=3)


def test_keywords_are_not_found_inside_other_words():
    assert match_keywords("Science Showcase", CLASSES)[0] != "toilet_or_wc"
    assert match_keywords("Product Design 01", CLASSES)[0] != "shaft_or_riser"
    assert match_keywords("WC's", CLASSES)[0] == "toilet_or_wc"
    assert match_keywords("(TypeB703)", CLASSES)[0] == "bedroom"        # OCR glues words: long keywords still match inside
    assert match_keywords("ScienceLab03", CLASSES)[0] == "classroom_or_lab"
    assert match_keywords("Sports Hall", CLASSES)[0] == "sports_or_assembly_hall"


def test_classification_is_scale_invariant():
    for ratio in (100, 200):
        pm = 28.346 * 100 / ratio
        pt2_per_m2 = pm ** 2
        side = 2.0 * pm                                               # a 2 m x 2 m cell with a round fixture
        poly = Polygon([(0, 0), (side, 0), (side, side), (0, side)])
        info = classify_room(poly, [], 1, CLASSES, pt2_per_m2)
        assert info["type"] == "bathroom" and info["approx_area_m2"] == pytest.approx(4.0, abs=0.1)


def test_circle_elements_find_circles_not_rectangles_and_scale_with_the_sheet():
    t = np.linspace(0, 2 * np.pi, 40)
    circ = np.c_[100 + 14 * np.cos(t), 100 + 14 * np.sin(t)]          # 1 m radius at 1:200 (14.17 pt per m)
    rect = np.array([[200, 100], [230, 100], [230, 120], [200, 120], [200, 100]], float)
    P = [VPath(circ, True, 0.2, None, None), VPath(rect, True, 0.2, None, None)]
    els = circle_elements(P, 0.5, 0.3, (0, 0, 400, 400))
    assert len(els) == 1 and els[0]["radius_m"] == pytest.approx(0.99, abs=0.05)


def test_gap_closing_handles_diagonal_walls():
    m = np.zeros((300, 300), np.uint8)
    cv2.line(m, (40, 260), (240, 60), 255, 7)                        # long 45 degree wall ...
    cv2.line(m, (270, 20), (270, 280), 255, 7)                       # ... and a vertical wall it does not reach (gap ~ 30 px)
    closed, lines = close_wall_gaps(m, 7, 60 / 0.11, door_frac=0.11, pair_frac=0.11, rectilinear_only=False)
    assert len(lines) >= 1
    assert max(l["p1"][0] for l in lines) > 255                      # the extension reaches the vertical wall (x = 270 - half thickness)
    _, none = close_wall_gaps(m, 7, 60 / 0.11, door_frac=0.11, pair_frac=0.11, rectilinear_only=True)
    assert len(none) == 0                                            # an axis-only extender cannot close it


def test_profile_deep_merge_keeps_unrelated_keys():
    base = {"vector": {"a": 1, "gap_closing": {"enabled": True, "max_gap_m": 3.0}}, "x": 1}
    out = deep_merge(base, {"vector": {"gap_closing": {"enabled": False}, "roi": [0, 0, 1, 1]}})
    assert out["vector"] == {"a": 1, "gap_closing": {"enabled": False, "max_gap_m": 3.0}, "roi": [0, 0, 1, 1]} and out["x"] == 1
    assert base["vector"]["gap_closing"]["enabled"] is True            # inputs are not mutated
