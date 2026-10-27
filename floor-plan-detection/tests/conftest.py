"""Shared fixtures. The raster integration fixtures run the real pipeline once per session on CPU (OCR included)."""
from pathlib import Path
import cv2
import numpy as np
import pymupdf
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT.parent


def _run(pdf, out):
    from src.pipeline import run_vector
    cfg = yaml.safe_load(open(ROOT / "config" / "inference.yaml"))
    cfg["compute"] = {"device": "cpu"}
    classes = yaml.safe_load(open(ROOT / "config" / "classes.yaml"))["classes"]
    return run_vector(str(pdf), cfg, classes, out, log=lambda *a: None)


@pytest.fixture(scope="session")
def fp1_out(tmp_path_factory):
    pdf = PDF_DIR / "floor_plan.pdf"
    if not pdf.exists():
        pytest.skip("floor_plan.pdf not present")
    out = tmp_path_factory.mktemp("fp1")
    _run(pdf, out)
    return out


@pytest.fixture(scope="session")
def fp2_out(tmp_path_factory):
    pdf = PDF_DIR / "floor_plan2.pdf"
    if not pdf.exists():
        pytest.skip("floor_plan2.pdf not present")
    out = tmp_path_factory.mktemp("fp2")
    _run(pdf, out)
    return out


# ---- synthetic raster plan with exactly known room rectangles -------------------------------------------------------
# image 800x600 px. Walls are 12 px thick. Interior (free) rectangles between wall faces (x0, y0, x1, y1), half-open:
#   A: left room        B: top-right room       C: bottom-right room
# Doors: A<->B at x=394..406 (y 100..160), B<->C at y=294..306 (x 500..560), A<->outside none.
SYN_W, SYN_H = 800, 600
SYN_ROOMS = {"A": (56, 56, 394, 544), "B": (406, 56, 744, 294), "C": (406, 306, 744, 544)}
SYN_DOORS = [(394, 100, 406, 160), (500, 294, 560, 306)]          # openings (x0, y0, x1, y1) cut through the walls
SYN_LABELS = {"A": ("Bedroom", (110, 300)), "B": ("Kitchen", (500, 190)), "C": ("Living room", (470, 440))}


def synthetic_plan(with_text=True):
    g = np.full((SYN_H, SYN_W), 255, np.uint8)
    cv2.rectangle(g, (50, 50), (750, 550), 0, 12)                   # outer walls, 12 px thick (centre line on the rectangle)
    cv2.rectangle(g, (394, 56), (406, 544), 0, -1)                 # vertical partition x 394..406
    cv2.rectangle(g, (406, 294), (744, 306), 0, -1)                # horizontal partition y 294..306
    for (x0, y0, x1, y1) in SYN_DOORS:                             # door openings
        cv2.rectangle(g, (x0, y0), (x1 - 1, y1 - 1), 255, -1)
    cv2.rectangle(g, (60, 60), (62, 62), 0, -1)
    if with_text:
        for name, (txt, (x, y)) in SYN_LABELS.items():
            cv2.putText(g, txt, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.1, 0, 3, cv2.LINE_AA)          # bold glyphs, as in real plans
    return g
