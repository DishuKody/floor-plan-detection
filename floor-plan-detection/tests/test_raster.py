from pathlib import Path
import json
import numpy as np
import cv2
import pytest
import yaml
from shapely.geometry import Polygon
from src.raster.pipeline_raster import label_type, printed_area
from src.raster.segmentation import thick_walls

ROOT = Path(__file__).resolve().parent.parent
PDF2 = ROOT.parent / "floor_plan2.pdf"


@pytest.mark.parametrize("text,cls", [("Kitchenveranda", "balcony"), ("Covered Veranda", "balcony"), ("PORCH", "balcony"), ("WC", "toilet_or_wc"),
                                      ("Living room", "living_room"), ("Bedroom3", "bedroom"), ("BEDROOM", "bedroom"), ("Bath", "bathroom"),
                                      ("BATHROOM", "bathroom"), ("Store", "storage"), ("Cupboard", "storage"), ("Entrance", "corridor_or_hallway"),
                                      ("Kitchen", "kitchen"), ("DINING", "dining_room"), ("12.8m2", None), ("TYPICAL FLOOR PLAN", None), ("info@maramani.com", None)])
def test_label_priority_and_non_room_text(text, cls):
    assert label_type(text) == cls          # 'Kitchen veranda' must be a balcony, not a kitchen


def test_printed_area_parsing():
    assert printed_area("12.8m2") == 12.8 and printed_area("18.3 m²") == 18.3
    assert printed_area("3.0M X 2.0M") == pytest.approx(6.0)
    assert printed_area("Bedroom3") is None


def test_hatched_wall_survives_but_thin_text_lines_do_not():
    g = np.full((300, 400), 255, np.uint8)
    g[100:112, 20:380] = 150                              # 12px wall
    g[100:112:2, 20:380] = 90                             # dithered texture like floor_plan.pdf
    cv2.line(g, (20, 200), (380, 200), 0, 2)              # 2px furniture / text-like line
    tw = thick_walls(g)
    walls = tw["walls"]
    assert walls[100:112, 40:360].mean() > 0.9
    assert walls[190:210, :].sum() == 0


@pytest.mark.skipif(not PDF2.exists(), reason="floor_plan2.pdf not present")
def test_raster_pdf_end_to_end_full_page(tmp_path):
    from src.pipeline import run_vector
    cfg = yaml.safe_load(open(ROOT / "config" / "inference.yaml"))
    cfg["compute"] = {"device": "cpu"}
    classes = yaml.safe_load(open(ROOT / "config" / "classes.yaml"))["classes"]
    rooms, elements, stats = run_vector(str(PDF2), cfg, classes, tmp_path, log=lambda *a: None)
    assert stats["mode"] == "raster"
    ov, ref, mk = (cv2.imread(str(tmp_path / n)) for n in ("overlay.png", "debug/render_reference.png", "room_masks.png"))
    assert ov.shape == ref.shape == mk.shape                                  # whole page, no crop / stretch
    types = sorted(r["type"] for r in rooms if r["type"] != "unknown")           # + at most a couple of flagged unlabelled enclosed shapes (e.g. a sink unit)
    assert types == sorted(["bedroom", "bedroom", "bathroom", "kitchen", "dining_room", "living_room", "balcony"])
    assert sum(r["type"] == "unknown" for r in rooms) <= 2
    for r in rooms:
        assert Polygon(r["polygon"]).is_valid and 0 <= np.array(r["polygon"]).min() and np.array(r["polygon"])[:, 0].max() <= ref.shape[1]
        assert len(r["polygon_image_px"]) >= 3                                # native-image frame is kept as well
    J = json.load(open(tmp_path / "rooms.json"))
    assert J["image"]["source_image"]["width"] == 600 and J["image"]["output_canvas"] == "full_page"
