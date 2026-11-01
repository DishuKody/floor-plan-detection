"""v7 evaluation against the hand-annotated A1800 image: red-style overlay, side-by-side with matched / missed / extra boundaries, room-level IoU,
and regression checks.
Usage (from floor-plan-detection): python -m src.evaluation.run_v7_evaluation [--run ../output_v7] [--base ../output_v6] [--out ../evaluation_v7]"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from . import annotated_regions as AR
from . import expected_ref as E
from .vector_regression import compare

ROOT = Path(__file__).resolve().parents[2]
PDFS = ROOT.parent


def banner(im, text, h=46):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (im.shape[1], h), (255, 255, 255), -1)
    cv2.putText(im, text, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(PDFS / "output_v7"))
    ap.add_argument("--base", default=str(PDFS / "output_v6"))
    ap.add_argument("--out", default=str(PDFS / "evaluation_v7"))
    ap.add_argument("--annotated", default=str(PDFS / "A1800 Ground Flr_expected.png"))
    a = ap.parse_args()
    run, base, out = Path(a.run), Path(a.base), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pdf = str(PDFS / "A1800 Ground Flr.pdf")
    S, ims = {}, {}
    for tag, d in (("v6", base / "A1800_Ground_Flr"), ("v7", run / "A1800_Ground_Flr")):
        sm, rows, im = AR.evaluate(pdf, d, a.annotated)
        ims[tag] = (rows, im, d)
        rooms = json.load(open(d / "rooms.json"))["rooms"]
        shape = im["target"].shape
        agg, _ = E.evaluate_lines(im["marks"], im["A"], rooms, shape, (8, 16, 24))
        sep = E.evaluate_separation(im["marks"], im["A"], rooms, shape, offset_px=10)
        S[f"a1800_{tag}"] = dict(room_level=sm, rows=rows, distance=agg,
                                 separation=dict(separated=sep["separated"], separated_both_in_rooms=sep["separated_both_in_rooms"]))

    rows, im, d = ims["v7"]
    rooms = json.load(open(d / "rooms.json"))["rooms"]
    target, A = im["target"], im["A"]
    # 1. our boundaries in the reference style: thick red on the plain page (full page, never cropped)
    red_only = target.copy()
    red_only[E.predicted_boundary_mask(rooms, target.shape, thickness=9) > 0] = (0, 0, 255)
    cv2.imwrite(str(out / "A1800_detected_boundaries_red_FULLPAGE.png"), red_only)
    # 2. matched / missed / extra
    TOL = 10
    pred1 = E.predicted_boundary_mask(rooms, target.shape, thickness=1)
    dt_pred = cv2.distanceTransform(255 - pred1, cv2.DIST_L2, 3)
    exp_line = im["red"] > 0
    dt_exp = cv2.distanceTransform((255 * (~exp_line)).astype(np.uint8), cv2.DIST_L2, 3)
    cmp = (target.astype(np.float32) * 0.45 + 255 * 0.55).astype(np.uint8)
    extra = (E.predicted_boundary_mask(rooms, target.shape, thickness=5) > 0) & (dt_exp > TOL)
    xs, ys = np.nonzero(exp_line.any(0))[0], np.nonzero(exp_line.any(1))[0]
    box = np.zeros_like(exp_line)
    box[max(int(ys.min()) - 40, 0):int(ys.max()) + 40, max(int(xs.min()) - 40, 0):int(xs.max()) + 40] = True
    extra &= box
    cmp[extra] = (255, 120, 0)                                                 # blue: detected, no annotation nearby (extra, or simply not annotated)
    matched, missed = exp_line & (dt_pred <= TOL), exp_line & (dt_pred > TOL)
    k5 = np.ones((5, 5), np.uint8)
    cmp[cv2.dilate(matched.astype(np.uint8), k5) > 0] = (0, 170, 0)           # green: expected AND detected
    cmp[cv2.dilate(missed.astype(np.uint8), k5) > 0] = (0, 0, 255)            # red: expected, MISSED
    S["a1800_v7"]["line_pixels"] = dict(matched=int(matched.sum()), missed=int(missed.sum()), tol_px=TOL,
                                        match_share=round(float(matched.sum() / max(exp_line.sum(), 1)), 3), extra_boundary_pixels_in_annotated_area=int(extra.sum()))
    cv2.imwrite(str(out / "A1800_matched_missed_extra_FULLPAGE.png"), cmp)
    # side by side in the frame of the user's screenshot (mapped back through the registration)
    exp_img = im["exp"]
    h0, w0 = exp_img.shape[:2]

    def in_ref_frame(img):
        return cv2.warpAffine(img, A.astype(np.float64), (w0, h0), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_AREA, borderValue=(255, 255, 255))
    cv2.imwrite(str(out / "A1800_side_by_side_annotation_detected_comparison.png"),
                np.vstack([banner(exp_img, "1  your annotation (reference)"), banner(in_ref_frame(red_only), "2  detected room boundaries (red)"),
                           banner(in_ref_frame(cmp), "3  GREEN = matched, RED = missed, BLUE = detected but not annotated / extra")]))
    areas_png = run / "A1800_Ground_Flr" / "areas_overlay.png"
    if areas_png.exists():
        cv2.imwrite(str(out / "A1800_areas_FULLPAGE.png"), cv2.imread(str(areas_png)))
        cv2.imwrite(str(out / "A1800_annotation_vs_independent_areas.png"),
                    np.vstack([banner(exp_img, "your annotation (reference)"), banner(in_ref_frame(cv2.imread(str(areas_png))), "detected independent areas: unique ID + distinct colour")]))
    # per-room IoU map
    lab, heat = im["lab"], target.copy()
    ov = heat.copy()
    for r in rows:
        x0, y0, x1, y1 = r["bbox"]
        reg = lab[y0:y1, x0:x1] == r["label"]
        ov[y0:y1, x0:x1][reg] = (0, 190, 0) if r["iou"] >= 0.7 else ((0, 200, 255) if r["iou"] >= 0.4 else (0, 0, 255))
    heat = cv2.addWeighted(heat, 0.55, ov, 0.45, 0)
    heat[im["red"] > 0] = (0, 0, 0)
    cv2.imwrite(str(out / "A1800_room_iou_map_FULLPAGE.png"), heat)
    cv2.imwrite(str(out / "A1800_v7_overlay_FULLPAGE.png"), cv2.imread(str(run / "A1800_Ground_Flr" / "overlay.png")))

    # ---- regressions
    S["student_vs_v4_deliverable"] = compare(ROOT / "outputs" / "Student_Accomodation", run / "Student_Accomodation")
    fp = {}
    for n in ("floor_plan", "floor_plan2"):
        A_ = json.load(open(base / n / "rooms.json"))["rooms"]
        B_ = json.load(open(run / n / "rooms.json"))["rooms"]
        fp[n] = dict(rooms=[len(A_), len(B_)], polygons_identical=[r["polygon"] for r in A_] == [r["polygon"] for r in B_],
                     types_identical=[r["type"] for r in A_] == [r["type"] for r in B_])
    S["raster_plans_vs_v6"] = fp
    S["floor_plan_expected_png"] = E.run(PDFS / "expected.png", run / "floor_plan", "v7", save_dir=out / "floor_plan_expected_png")["aggregate"]
    for n in ("A1800_Ground_Flr", "Student_Accomodation", "floor_plan", "floor_plan2"):
        for suf in ("overlay.png", "masks.png", "rooms.json", "elements.json"):
            (out / f"{n}_{suf}").write_bytes((run / n / f"{n}_{suf}").read_bytes())
    json.dump(S, open(out / "summary.json", "w"), indent=1, default=str)
    for t in ("v6", "v7"):
        print(t, S[f"a1800_{t}"]["room_level"], {k: S[f"a1800_{t}"]["distance"][k] for k in ("mean_dist_px", "pixel_recall@8px")}, S[f"a1800_{t}"]["separation"])
    print("lines v7:", S["a1800_v7"]["line_pixels"])
    print("student:", {k: S["student_vs_v4_deliverable"][k] for k in ("rooms_new", "same_type", "min_iou", "max_vertex_displacement_pt")})
    print("raster:", fp)
    print("expected.png:", {k: S["floor_plan_expected_png"][k] for k in ("mean_dist_px", "pixel_recall@8px", "lines_with_recall>=0.8@8px")})


if __name__ == "__main__":
    main()
