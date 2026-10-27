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
        for ex in r.get("extra_polygons", []):
            cv2.fillPoly(m, [(np.array(ex) - [x0, y0]).astype(np.int32)], 255)
        for h in r.get("holes", []):
            cv2.fillPoly(m, [(np.array(h) - [x0, y0]).astype(np.int32)], 0)
        ov[m > 0] = col
    img = cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)
    for r in rooms:
        col = tuple(int(c) for c in classes[r["type"]]["color"][::-1])
        for poly_px in [r["polygon"]] + list(r.get("extra_polygons", [])):
            cv2.polylines(img, [(np.array(poly_px) - [x0, y0]).astype(np.int32)], True, tuple(int(c * 0.6) for c in col), 2, cv2.LINE_AA)
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


def add_legend_inset(img, classes, used, extra_lines=(), candidates=None, margin=30):
    """Legend drawn INSIDE the full-page canvas (no canvas growth). Its size scales with the page, and its position is the
    emptiest spot found on a coarse grid (corners preferred on ties), so the plan is never obscured. Returns (img, info)."""
    names = [n for n in classes if n in used]
    lines = [(f"{n} ({used[n]})", tuple(int(c) for c in classes[n]["color"][::-1])) for n in names] + [(t, c) for t, c in extra_lines]
    H, W = img.shape[:2]
    sc = max(0.6, min(1.0, W / 2400.0))
    rowh, w = int(30 * sc), int(380 * sc)
    h = len(lines) * rowh + int(24 * sc)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = (gray < 235).astype(np.uint8)                      # anything that is not (near) white paper counts as content
    integ = cv2.integral(ink)
    best = None
    step = max(10, int(20 * sc))
    for y in range(margin, max(margin + 1, H - h - margin + 1), step):
        for x in range(margin, max(margin + 1, W - w - margin + 1), step):
            n = integ[y + h, x + w] - integ[y, x + w] - integ[y + h, x] + integ[y, x]
            corner = min(x, W - w - x) + min(y, H - h - y)    # tie-break: prefer a corner
            key = (n, corner)
            if best is None or key < best[0]:
                best = (key, x, y, n)
    (_, _), x, y, n = best
    frac = n / float(h * w)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)
    yy = y + int(24 * sc)
    for txt, col in lines:
        cv2.rectangle(img, (x + int(12 * sc), yy - int(14 * sc)), (x + int(38 * sc), yy + int(6 * sc)), col, -1)
        cv2.putText(img, txt, (x + int(50 * sc), yy + int(4 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * sc, (30, 30, 30), 1, cv2.LINE_AA)
        yy += rowh
    return img, dict(anchor="grid_search_emptiest", box_px=[x, y, x + w, y + h], ink_fraction_under_legend=round(float(frac), 4))
