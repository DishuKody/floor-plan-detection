"""Evaluation against `expected.png` (a hand-annotated screenshot of an earlier overlay of floor_plan.pdf).

What expected.png IS: a 644x898 screenshot of our previous overlay, plus red marks drawn by the user:
  * ~13 thick (3-5 px) straight red LINES = where a room boundary is expected (walls / door thresholds),
  * 2 large red X's over the title text and the logo = "this is not a room / shape".
Our own review rings / door dots are also red but small (<60 px area) and are discarded by size.
It is SPARSE: it marks expected boundaries only at some places, so recall of those lines is measurable, precision is not.

Pipeline: red-mark extraction -> registration (SIFT + RANSAC affine, marks inpainted) of the screenshot to an output overlay
-> metrics (distance of every expected-line pixel to the nearest predicted room boundary; recall at tolerances).
The reference image is never modified; all outputs are separate files."""
import json
import sys
from pathlib import Path
import numpy as np
import cv2
from skimage.morphology import skeletonize

TOLS_PX = (4, 8, 12)        # in overlay pixels (scale 2.0 px/pt => 2, 4, 6 pt); the hand-drawn lines are only accurate to a few px


def red_mask(bgr):
    b, g, r = [bgr[:, :, i].astype(int) for i in range(3)]
    return ((r > 170) & (g < 110) & (b < 110) & (r - g > 90)).astype(np.uint8)


def _skeleton_topology(comp):
    sk = skeletonize(comp)
    nb = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk.astype(np.uint8)
    jn = (sk & (nb >= 3)).astype(np.uint8)
    n_j = cv2.connectedComponents(jn, connectivity=8)[0] - 1
    return sk, int((sk & (nb == 1)).sum()), int(n_j)


def extract_marks(bgr, min_area=60, x_area=800, legend_box=None, networks=False, min_side=20):
    """-> dict(lines=[{'pts': Nx2 skeleton pixels (x,y) in expected.png px, 'bbox':..}], x_marks=[bbox], mask=uint8).
    networks=True (thick strokes that join into networks, e.g. a Paint annotation on a large sheet): every red component with a side >= min_side px is a
    line network; a plain cross (skeleton with exactly 4 ends and at most two junctions, i.e. one hand-drawn crossing) is a 'not a shape' X mark; rings / dots are discarded."""
    m = red_mask(bgr)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    lines, xs, keep = [], [], np.zeros_like(m)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_area:
            continue                                   # our own rings / dots / legend swatches
        if legend_box and legend_box[0] <= x <= legend_box[2] and legend_box[1] <= y <= legend_box[3]:
            continue
        comp = (lab == i)
        if networks:
            if max(w, h) < min_side:
                continue                                   # review rings, dots, legend swatches
            keep[comp] = 1
            sk, ends, n_j = _skeleton_topology(comp)
            if ends == 4 and n_j <= 2:
                xs.append([int(x), int(y), int(x + w), int(y + h)])
            else:
                yy, xx = np.nonzero(sk)
                lines.append(dict(pts=np.c_[xx, yy].astype(float), bbox=[int(x), int(y), int(x + w), int(y + h)], area=int(a)))
            continue
        keep[comp] = 1
        if a >= x_area:
            xs.append([int(x), int(y), int(x + w), int(y + h)])
            continue
        sk = skeletonize(comp)
        yy, xx = np.nonzero(sk)
        lines.append(dict(pts=np.c_[xx, yy].astype(float), bbox=[int(x), int(y), int(x + w), int(y + h)], area=int(a)))
    return dict(lines=lines, x_marks=xs, mask=keep)


def register(expected_bgr, target_bgr, marks_mask, scale_hint=None):
    """Affine (6 dof) mapping expected.png px -> target overlay px. Red marks are inpainted before matching."""
    clean = cv2.inpaint(expected_bgr, cv2.dilate(marks_mask, np.ones((5, 5), np.uint8)), 3, cv2.INPAINT_TELEA)
    g1 = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    s = scale_hint or g2.shape[1] / g1.shape[1]       # scale_hint = overlay px per screenshot px when the screenshot is a crop of the page
    g1u = cv2.resize(g1, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)       # same rough scale helps SIFT
    sift = cv2.SIFT_create(4000)
    k1, d1 = sift.detectAndCompute(g1u, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    bf = cv2.BFMatcher()
    good = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < 0.7 * n.distance]
    src = np.float32([k1[m.queryIdx].pt for m in good]) / s                        # back to expected.png px
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    A, inl = cv2.estimateAffine2D(src, dst, ransacReprojThreshold=2.5, maxIters=5000, confidence=0.999)
    inl = inl.ravel().astype(bool)
    res = np.linalg.norm((src[inl] @ A[:, :2].T + A[:, 2]) - dst[inl], axis=1)
    return A, dict(n_matches=len(good), n_inliers=int(inl.sum()), rms_px=round(float(np.sqrt((res ** 2).mean())), 2),
                   median_px=round(float(np.median(res)), 2), max_px=round(float(res.max()), 2),
                   scale_x=round(float(np.hypot(A[0, 0], A[1, 0])), 4), scale_y=round(float(np.hypot(A[0, 1], A[1, 1])), 4))


def predicted_boundary_mask(rooms, shape, scale_key="polygon", thickness=1):
    m = np.zeros(shape[:2], np.uint8)
    for r in rooms:
        for poly in [r[scale_key]] + list(r.get("extra_polygons", [])):
            cv2.polylines(m, [np.round(np.array(poly)).astype(np.int32)], True, 255, thickness)
        for h in r.get("holes", []):
            cv2.polylines(m, [np.round(np.array(h)).astype(np.int32)], True, 255, thickness)
    return m


def evaluate_lines(marks, A, rooms, shape, tols=TOLS_PX):
    pred = predicted_boundary_mask(rooms, shape)
    dt = cv2.distanceTransform(255 - pred, cv2.DIST_L2, 3)
    per, allpts = [], []
    for k, ln in enumerate(marks["lines"]):
        pts = ln["pts"] @ A[:, :2].T + A[:, 2]
        px = np.clip(np.round(pts).astype(int), 0, [shape[1] - 1, shape[0] - 1])
        d = dt[px[:, 1], px[:, 0]]
        allpts.append(d)
        per.append(dict(id=k, length_px=round(float(len(d)), 1), mean_dist_px=round(float(d.mean()), 2), median_dist_px=round(float(np.median(d)), 2),
                        max_dist_px=round(float(d.max()), 2), **{f"recall@{t}px": round(float((d <= t).mean()), 3) for t in tols},
                        bbox_expected_png=ln["bbox"], centre_overlay_px=[round(float(pts[:, 0].mean()), 1), round(float(pts[:, 1].mean()), 1)]))
    d = np.concatenate(allpts)
    agg = dict(n_lines=len(per), n_px=int(len(d)), mean_dist_px=round(float(d.mean()), 2), median_dist_px=round(float(np.median(d)), 2),
               p90_dist_px=round(float(np.percentile(d, 90)), 2), max_dist_px=round(float(d.max()), 2),
               **{f"pixel_recall@{t}px": round(float((d <= t).mean()), 3) for t in tols},
               **{f"lines_with_recall>=0.8@{t}px": sum(p[f"recall@{t}px"] >= 0.8 for p in per) for t in tols})
    return agg, per


def room_id_map(rooms, shape):
    """int32 map of the page: 1 + room index inside a room polygon (extra polygons included, holes cut out), 0 elsewhere."""
    m = np.zeros(shape[:2], np.int32)
    for i, r in enumerate(rooms, 1):
        for poly in [r["polygon"]] + list(r.get("extra_polygons", [])):
            cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], i)
        for h in r.get("holes", []):
            cv2.fillPoly(m, [np.round(np.array(h)).astype(np.int32)], 0)
    return m


def evaluate_separation(marks, A, rooms, shape, offset_px=12, k_tangent=9):
    """What an annotated room boundary MEANS: the two sides of the line belong to different rooms. Distance recall cannot tell (a dense mesh of
    polygons is near every line); this does. For every skeleton pixel the local normal is estimated from its nearest skeleton neighbours and the
    room ids at +-offset_px are compared. `separated` = ids differ; `separated_both_in_rooms` additionally requires both sides to lie in a room
    (a wall band between two rooms is then not counted). offset_px must exceed the hand-drawing error of the line."""
    ids = room_id_map(rooms, shape)
    per, tot, sep, sep2 = [], 0, 0, 0
    for k, ln in enumerate(marks["lines"]):
        pts = ln["pts"] @ A[:, :2].T + A[:, 2]
        if len(pts) < 3:
            continue
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        nn = np.argsort(d2, axis=1)[:, :min(k_tangent, len(pts))]
        a = b = 0
        for i in range(len(pts)):
            q = pts[nn[i]] - pts[nn[i]].mean(0)
            t = np.linalg.svd(q, full_matrices=False)[2][0]
            n = np.array([-t[1], t[0]])
            side = [np.clip(np.round(pts[i] + sg * offset_px * n).astype(int), 0, [shape[1] - 1, shape[0] - 1]) for sg in (1, -1)]
            i1, i2 = int(ids[side[0][1], side[0][0]]), int(ids[side[1][1], side[1][0]])
            a += i1 != i2
            b += (i1 != i2) and i1 > 0 and i2 > 0
        per.append(dict(id=k, n=len(pts), separated=round(a / len(pts), 3), separated_both_in_rooms=round(b / len(pts), 3)))
        tot += len(pts); sep += a; sep2 += b
    return dict(offset_px=offset_px, separated=round(sep / max(tot, 1), 3), separated_both_in_rooms=round(sep2 / max(tot, 1), 3), per_line=per)


def evaluate_x_marks(marks, A, rooms, shape):
    """The X marks say 'no shape here': report the largest room-polygon area covering each marked region."""
    out = []
    for bb in marks["x_marks"]:
        c = np.array([[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]], float) @ A[:, :2].T + A[:, 2]
        box = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(box, [np.round(c).astype(np.int32)], 1)
        cover = 0.0
        for r in rooms:
            m = np.zeros(shape[:2], np.uint8)
            cv2.fillPoly(m, [np.round(np.array(r["polygon"])).astype(np.int32)], 1)
            cover = max(cover, float((m & box).sum()) / max(box.sum(), 1))
        out.append(dict(region_overlay_px=np.round(c, 1).tolist(), max_room_polygon_coverage=round(cover, 4), ok_no_shape=bool(cover < 0.02)))
    return out


def run(expected_png, out_dir, tag=None, scale_key="polygon", save_dir=None, register_overlay=None, networks=False, tols=TOLS_PX, scale_hint=None, sep_offset_px=6):
    exp = cv2.imread(str(expected_png), cv2.IMREAD_COLOR)
    out_dir = Path(out_dir)
    J = json.load(open(out_dir / "rooms.json"))
    rooms = J["rooms"]
    target = cv2.imread(str(out_dir / "debug" / "render_reference.png"))             # plain page render = overlay canvas
    ov = cv2.imread(str(out_dir / "overlay.png"))
    legend = J["image"].get("legend", {}).get("box_px")
    # the screenshot was taken from an OVERLAY, so register against the overlay (it contains the same coloured fills)
    marks = extract_marks(exp, legend_box=None, networks=networks)
    # register against the overlay the screenshot was taken from (same for every version compared), not against the version under test
    reg_target = cv2.imread(str(register_overlay)) if register_overlay else ov
    A, reg = register(exp, reg_target, marks["mask"], scale_hint)
    agg, per = evaluate_lines(marks, A, rooms, ov.shape, tols)
    xs = evaluate_x_marks(marks, A, rooms, ov.shape)
    sepm = evaluate_separation(marks, A, rooms, ov.shape, offset_px=sep_offset_px)
    res = dict(tag=tag or str(out_dir), registration=reg, aggregate=agg, separation=sepm, lines=per, x_marks=xs, affine_expected_to_overlay_px=A.round(5).tolist())
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(save_dir / "expected_eval.json", "w"), indent=1)
        draw_comparison(exp, marks, A, rooms, ov, target, save_dir, tag or "")
    return res


def draw_comparison(exp, marks, A, rooms, ov, plain, save_dir, tag):
    """Full-page images (never cropped): the plain page with expected lines in RED and detected boundaries in BLUE, plus side-by-side."""
    H, W = plain.shape[:2]
    base = plain.copy()
    pred = predicted_boundary_mask(rooms, base.shape, thickness=2)
    base[pred > 0] = (255, 120, 0)                                                   # detected boundaries: blue-ish (BGR)
    for ln in marks["lines"]:
        p = ln["pts"] @ A[:, :2].T + A[:, 2]
        for x, y in np.round(p).astype(int):
            cv2.circle(base, (x, y), 2, (0, 0, 255), -1)                              # expected: red
    for bb in marks["x_marks"]:
        c = np.array([[bb[0], bb[1]], [bb[2], bb[3]], [bb[2], bb[1]], [bb[0], bb[3]]], float) @ A[:, :2].T + A[:, 2]
        cv2.line(base, tuple(np.round(c[0]).astype(int)), tuple(np.round(c[1]).astype(int)), (0, 0, 255), 3)
        cv2.line(base, tuple(np.round(c[2]).astype(int)), tuple(np.round(c[3]).astype(int)), (0, 0, 255), 3)
    cv2.imwrite(str(save_dir / "expected_vs_detected_fullpage.png"), base)
    warped = cv2.warpAffine(exp, A, (W, H), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    cv2.imwrite(str(save_dir / "expected_warped_to_page.png"), warped)
    side = np.hstack([warped, np.full((H, 12, 3), 0, np.uint8), base])
    cv2.imwrite(str(save_dir / "side_by_side_expected_warped__vs__detected.png"), side)


if __name__ == "__main__":
    ref, d, tag = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else None)
    sd = sys.argv[4] if len(sys.argv) > 4 else None
    r = run(ref, d, tag, save_dir=sd)
    print(json.dumps(dict(registration=r["registration"], aggregate=r["aggregate"], x_marks=r["x_marks"]), indent=1))
    for p in r["lines"]:
        print(f"  line {p['id']:2d} len {p['length_px']:6.1f} mean {p['mean_dist_px']:5.1f} max {p['max_dist_px']:5.1f}  recall@4 {p['recall@4px']:.2f} @8 {p['recall@8px']:.2f} @12 {p['recall@12px']:.2f}  at {p['centre_overlay_px']}")
