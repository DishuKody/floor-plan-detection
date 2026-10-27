import numpy as np
import cv2


def _poly_px(pts, scale, ox=0, oy=0):
    return (np.array(pts) * scale - [ox, oy]).astype(np.int32)


def draw_overlay(base_bgr, rooms, classes, elements, crop_pt, scale, alpha=0.38):
    """base_bgr: render at `scale` px/pt. crop_pt: [x0,y0,x1,y1] in points. rooms use polygon_pt/holes via polygon (px)."""
    x0, y0, x1, y1 = [int(v * scale) for v in crop_pt]
    img = base_bgr[y0:y1, x0:x1].copy()
    ov = img.copy()
    for r in rooms:
        col = tuple(int(c) for c in classes[r["type"]]["color"][::-1])
        poly = np.array(r["polygon"]) - [x0, y0]
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(m, [poly.astype(np.int32)], 255)
        for h in r.get("holes", []):
            cv2.fillPoly(m, [(np.array(h) - [x0, y0]).astype(np.int32)], 0)
        ov[m > 0] = col
    img = cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)
    for r in rooms:
        col = tuple(int(c) for c in classes[r["type"]]["color"][::-1])
        cv2.polylines(img, [(np.array(r["polygon"]) - [x0, y0]).astype(np.int32)], True, tuple(int(c * 0.6) for c in col), 2, cv2.LINE_AA)
        if r.get("review_required"):
            cx, cy = (np.array(r["polygon"]).mean(0) - [x0, y0]).astype(int)
            cv2.circle(img, (cx, cy), 6, (0, 0, 255), 2, cv2.LINE_AA)
    for e in elements:
        if e["type"] == "door":
            c = (np.array(e["hinge_pt"]) * scale - [x0, y0]).astype(int)
            cv2.circle(img, tuple(c), 4, (0, 0, 220), -1, cv2.LINE_AA)
        elif e["type"] == "window":
            cv2.polylines(img, [(np.array(e["polygon_pt"]) * scale - [x0, y0]).astype(np.int32)], True, (230, 160, 0), 2, cv2.LINE_AA)
        elif e["type"] == "column":
            b = np.array(e["bbox_pt"]) * scale - [x0, y0, x0, y0]
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 60, 60), 2)
    return img


def add_legend(img, classes, used):
    names = [n for n in classes if n in used]
    pad, rowh = 14, 30
    w = 330
    legend = np.full((img.shape[0], w, 3), 255, np.uint8)
    y = pad + 10
    for n in names:
        col = tuple(int(c) for c in classes[n]["color"][::-1])
        cv2.rectangle(legend, (pad, y - 14), (pad + 26, y + 6), col, -1)
        cv2.putText(legend, f"{n} ({used[n]})", (pad + 38, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        y += rowh
    y += 10
    for txt, c in [("red ring = review required", (0, 0, 255)), ("red dot = door hinge", (0, 0, 220)), ("blue box = window", (230, 160, 0)), ("grey box = column", (60, 60, 60))]:
        cv2.putText(legend, txt, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA); y += 24
    return np.hstack([img, legend])
