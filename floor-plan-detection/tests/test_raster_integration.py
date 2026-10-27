"""End-to-end checks on the REAL pipeline (CPU, OCR included): expected.png reference, coordinates, full-page preservation, synthetic PDF."""
import json
from pathlib import Path
import cv2
import numpy as np
import pymupdf
import pytest
import yaml
from shapely.geometry import Polygon, box

from tests.conftest import (ROOT, PDF_DIR, SYN_ROOMS, SYN_LABELS, SYN_W, SYN_H, synthetic_plan, _run)

EXPECTED = PDF_DIR / "expected.png"
V3_OVERLAY = ROOT / "versions" / "v3" / "raster_output" / "overlay.png"
SHAPES = {"circle", "square", "rectangle", "rectilinear_polygon", "irregular_polygon"}


# ---------------------------------------------------------------------------------------------------- expected.png
@pytest.mark.skipif(not EXPECTED.exists(), reason="expected.png not present")
def test_boundaries_match_the_hand_drawn_expected_lines(fp1_out):
    """Thresholds sit just below the measured v4 values (mean 5.3 px, recall@8 84 %, 9/13 lines) and far from v3 (21.1 px, 40 %, 1/13)."""
    from src.evaluation import expected_ref as ER
    r = ER.run(EXPECTED, fp1_out, "v4", register_overlay=(V3_OVERLAY if V3_OVERLAY.exists() else None))
    assert r["registration"]["rms_px"] <= 1.0 and r["registration"]["n_inliers"] >= 300         # the screenshot is registered precisely
    a = r["aggregate"]
    assert a["n_lines"] == 13
    assert a["mean_dist_px"] <= 6.5, a
    assert a["pixel_recall@8px"] >= 0.80, a
    assert a["lines_with_recall>=0.8@8px"] >= 8, a


@pytest.mark.skipif(not EXPECTED.exists(), reason="expected.png not present")
def test_title_and_logo_are_not_detected_as_shapes(fp1_out):
    from src.evaluation import expected_ref as ER
    r = ER.run(EXPECTED, fp1_out, "v4", register_overlay=(V3_OVERLAY if V3_OVERLAY.exists() else None))
    assert len(r["x_marks"]) == 2 and all(x["ok_no_shape"] and x["max_room_polygon_coverage"] < 0.02 for x in r["x_marks"])


# ---------------------------------------------------------------------------------------------------- coordinates
@pytest.mark.parametrize("fx", ["fp1_out", "fp2_out"])
def test_polygon_coordinates_agree_across_the_three_frames(fx, request):
    out = request.getfixturevalue(fx)
    J = json.load(open(out / "rooms.json"))
    s = J["image"]["transform"]["scale_px_per_pt"]
    x0, y0, x1, y1 = J["image"]["source_image"]["placed_rect_pt"]
    W, H = J["image"]["source_image"]["width"], J["image"]["source_image"]["height"]
    for r in J["rooms"]:
        pg = np.array(r["polygon"])                    # page-render px
        pt = np.array(r["polygon_pt"])                 # PDF points
        im = np.array(r["polygon_image_px"])           # embedded-image px
        assert np.allclose(pg, pt * s, atol=0.06)
        back = im * [(x1 - x0) / W, (y1 - y0) / H] + [x0, y0]
        assert np.allclose(back, pt, atol=0.01)        # image px -> pt with the placed rectangle
        assert 0 <= pg[:, 0].min() and pg[:, 0].max() <= J["image"]["width"] and 0 <= pg[:, 1].min() and pg[:, 1].max() <= J["image"]["height"]


# ---------------------------------------------------------------------------------------------------- full page
@pytest.mark.parametrize("fx", ["fp1_out", "fp2_out"])
def test_raster_outputs_preserve_the_whole_page_exactly(fx, request):
    out = request.getfixturevalue(fx)
    J = json.load(open(out / "rooms.json"))
    E = json.load(open(out / "elements.json"))["elements"]
    ov, ref, mk = (cv2.imread(str(out / n)) for n in ("overlay.png", "debug/render_reference.png", "room_masks.png"))
    s = J["image"]["transform"]["scale_px_per_pt"]
    pw, ph = J["image"]["transform"]["page_size_pt"]
    assert ov.shape == ref.shape == mk.shape and abs(ov.shape[1] - pw * s) <= 1 and abs(ov.shape[0] - ph * s) <= 1        # whole page, same aspect
    painted = np.zeros(ov.shape[:2], np.uint8)
    for r in J["rooms"]:
        cv2.fillPoly(painted, [np.round(np.array(r["polygon"])).astype(np.int32)], 255)
    painted = cv2.dilate(painted, np.ones((9, 9), np.uint8))
    for e in E:
        if e["type"] == "door":
            cv2.circle(painted, tuple((np.array(e["hinge_pt"]) * s).astype(int)), 8, 255, -1)
        elif e["type"] == "circle":
            cv2.circle(painted, tuple((np.array(e["center_pt"]) * s).astype(int)), int(e["radius_pt"] * s) + 6, 255, -1)
        elif e["type"] == "opening_line":
            cv2.line(painted, tuple((np.array(e["p0_pt"]) * s).astype(int)), tuple((np.array(e["p1_pt"]) * s).astype(int)), 255, 9)
    lx0, ly0, lx1, ly1 = J["image"]["legend"]["box_px"]
    painted[ly0 - 2:ly1 + 3, lx0 - 2:lx1 + 3] = 255
    diff = (np.abs(ov.astype(int) - ref.astype(int)).sum(2) > 0) & (painted == 0)
    assert int(diff.sum()) == 0                                                  # nothing outside the paint differs from the plain page render
    assert (ov[:3] == ref[:3]).all() and (ov[-3:] == ref[-3:]).all() and (ov[:, :3] == ref[:, :3]).all() and (ov[:, -3:] == ref[:, -3:]).all()
    assert J["image"]["legend"]["ink_fraction_under_legend"] < 0.01              # the legend hides no drawing content


@pytest.mark.parametrize("fx", ["fp1_out", "fp2_out"])
def test_every_room_has_valid_shape_and_polygon(fx, request):
    out = request.getfixturevalue(fx)
    for r in json.load(open(out / "rooms.json"))["rooms"]:
        assert Polygon(r["polygon_image_px"], r["holes_image_px"]).is_valid
        assert r["shape"]["shape_type"] in SHAPES and r["shape"]["n_vertices"] >= 3
        assert not any("text" in f.lower() and "hole" in f.lower() for f in r["flags"])
        assert len(r["holes_image_px"]) == 0                                    # no text-induced holes


def test_vector_outputs_also_carry_shape_descriptors_without_changing_geometry():
    p = ROOT / "outputs" / "rooms.json"
    if not p.exists():
        pytest.skip("run the vector pipeline first")
    rooms = json.load(open(p))["rooms"]
    assert all(r["shape"]["shape_type"] in SHAPES for r in rooms)
    assert {r["shape"]["shape_type"] for r in rooms} & {"rectangle", "square", "rectilinear_polygon"}


# ---------------------------------------------------------------------------------------------------- synthetic PDF
def test_synthetic_pdf_rooms_land_on_the_known_rectangles_in_pdf_points(tmp_path):
    """Image 800x600 px embedded in a PDF at a known rectangle; the true room rectangles are known in image px, so the
    PDF-point output can be checked against ground truth through the whole transform chain (OCR included)."""
    img = synthetic_plan()
    ok, png = cv2.imencode(".png", img)
    rect = pymupdf.Rect(60, 120, 460, 420)                                       # 400 x 300 pt  (0.5 pt per image px)
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(rect, stream=png.tobytes())
    pdf = tmp_path / "synthetic.pdf"
    doc.save(str(pdf))
    rooms, elements, stats = _run(pdf, tmp_path / "out")
    assert stats["mode"] == "raster"
    got = {r["type"]: r for r in rooms}
    expect = {"bedroom": "A", "kitchen": "B", "living_room": "C"}
    assert set(expect) <= set(got), sorted(got)
    for typ, name in expect.items():
        x0, y0, x1, y1 = SYN_ROOMS[name]
        gt = box(60 + 0.5 * x0, 120 + 0.5 * y0, 60 + 0.5 * x1, 120 + 0.5 * y1)  # the true room in PDF points
        pg = Polygon(got[typ]["polygon_pt"])
        assert pg.intersection(gt).area / pg.union(gt).area >= 0.97, (typ, pg.bounds, gt.bounds)
        assert gt.exterior.hausdorff_distance(pg.exterior) <= 2.0                # within 2 pt (=4 image px) everywhere
    ov, ref = cv2.imread(str(tmp_path / "out" / "overlay.png")), cv2.imread(str(tmp_path / "out" / "debug" / "render_reference.png"))
    assert ov.shape == ref.shape and ref.shape[1] == round(595 * 2.0) and ref.shape[0] == round(842 * 2.0)
