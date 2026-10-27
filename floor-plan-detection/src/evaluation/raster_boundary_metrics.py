"""Reference-free boundary-quality metrics for raster-mode outputs (usable on plans that have no expected.png).

For every room boundary pixel (exteriors + holes, in embedded-image px) measured against ONE neutral mask set - the v4 wall mask
(text suppressed) and its wall-extension closures - applied identically to every version being compared:
  on_wall      boundary within `tol` px of a wall face
  on_closure   within `tol` px of a wall-extension line (door threshold / opening) but not of a wall
  in_free      neither: curved / partition / leaking boundary
  on_text      boundary inside an OCR text box (outlines hugging glyphs = clutter)
plus vertices per room and number of holes. Higher on_wall+on_closure and lower on_text/holes = cleaner outlines."""
import json
import sys
from pathlib import Path
import numpy as np
import cv2


def _to_image_px(poly_page, J):
    tr, src = J["image"]["transform"], J["image"].get("source_image")
    rect = src["placed_rect_pt"]
    W, H = src["width"], src["height"]
    sx, sy = (rect[2] - rect[0]) / W, (rect[3] - rect[1]) / H
    p = np.asarray(poly_page, float) / tr["scale_px_per_pt"]
    return (p - [rect[0], rect[1]]) / [sx, sy]


def boundary_masks(dir_):
    d = Path(dir_)
    J = json.load(open(d / "rooms.json"))
    W, H = J["image"]["source_image"]["width"], J["image"]["source_image"]["height"]
    m = np.zeros((H, W), np.uint8)
    nv = nh = 0
    for r in J["rooms"]:
        ext = _to_image_px(r["polygon"], J)
        cv2.polylines(m, [np.round(ext).astype(np.int32)], True, 255, 1)
        nv += len(ext)
        for h in r.get("holes", []):
            cv2.polylines(m, [np.round(_to_image_px(h, J)).astype(np.int32)], True, 255, 1)
            nh += 1
    return J, m, nv, nh


def metrics(dir_, ref_dir, tol=2.5):
    ref = Path(ref_dir)
    J, b, nv, nh = boundary_masks(dir_)
    walls = cv2.imread(str(ref / "debug" / "wall_mask_native.png"), 0) < 128
    closed = cv2.imread(str(ref / "debug" / "wall_mask_closed_native.png"), 0) < 128
    closure = closed & ~walls
    texts = json.load(open(ref / "debug" / "ocr.json"))
    tm = np.zeros(b.shape, np.uint8)
    for t in texts:
        x0, y0, x1, y1 = [int(round(v)) for v in t["bbox"]]
        tm[max(0, y0 - 1):y1 + 2, max(0, x0 - 1):x1 + 2] = 1
    dw = cv2.distanceTransform((~walls).astype(np.uint8), cv2.DIST_L2, 3)
    dc = cv2.distanceTransform((~closure).astype(np.uint8), cv2.DIST_L2, 3)
    ys, xs = np.nonzero(b)
    n = len(ys)
    on_w = dw[ys, xs] <= tol
    on_c = (~on_w) & (dc[ys, xs] <= tol)
    in_free = ~(on_w | on_c)
    on_t = tm[ys, xs] > 0
    return dict(n_rooms=len(J["rooms"]), boundary_px=int(n), on_wall=round(float(on_w.mean()), 3), on_closure=round(float(on_c.mean()), 3),
                in_free_space=round(float(in_free.mean()), 3), on_text=round(float(on_t.mean()), 3),
                vertices_per_room=round(nv / max(len(J["rooms"]), 1), 1), n_holes=nh,
                mean_dist_to_nearest_wall_or_closure_px=round(float(np.minimum(dw[ys, xs], dc[ys, xs]).mean()), 2))


if __name__ == "__main__":
    ref = sys.argv[1]
    for d in sys.argv[2:]:
        print(d, json.dumps(metrics(d, ref)))
