"""Room-level comparison against a hand-annotated reference (synthetic: two rooms closed by a hand-drawn line on their open side)."""
import numpy as np
import cv2
from src.evaluation import annotated_regions as AR


def _scene():
    wall = np.zeros((200, 400), np.uint8)
    cv2.rectangle(wall, (20, 20), (380, 120), 255, 3)                           # a building outline: top, left, right and bottom walls
    cv2.line(wall, (200, 20), (200, 95), 255, 3)                                # partition that stops short of the bottom wall (open frontage)
    red = np.zeros_like(wall)
    cv2.line(red, (20, 95), (200, 95), 255, 3)                                  # hand-drawn frontage of the left room only
    return wall, red


def test_expected_regions_use_the_red_line_to_close_the_room():
    wall, red = _scene()
    cut = wall.copy()
    cv2.line(cut, (20, 118), (380, 118), 0, 5)                                  # open the bottom wall: without the red line the left room leaks away
    cv2.line(cut, (20, 20), (20, 118), 255, 3)
    lab, regs = AR.expected_regions(cut, red, [], px_per_m=10.0, area_m2=(1.0, 500.0), gap_px=3)
    assert len(regs) >= 1
    left = min(regs, key=lambda r: abs(r["centroid"][0] - 110))
    assert 70 < left["centroid"][0] < 150 and left["centroid"][1] < 95           # the pocket above the red line


def test_compare_scores_split_and_merged_detections_low():
    lab = np.zeros((100, 200), np.int32)
    lab[10:90, 10:90] = 1
    lab[10:90, 110:190] = 2
    regs = [dict(label=1, bbox=[10, 10, 90, 90]), dict(label=2, bbox=[110, 10, 190, 90])]
    ids = np.zeros_like(lab)
    ids[10:90, 10:90] = 1                                                        # room 1 detected exactly
    ids[10:50, 110:190] = 2                                                      # room 2 detected as two halves
    ids[50:90, 110:190] = 3
    rows = AR.compare(lab, regs, ids, 3)
    assert rows[0]["iou"] > 0.99 and 0.45 < rows[1]["iou"] < 0.55
    assert AR.summarize(rows)["n_expected_rooms"] == 2
