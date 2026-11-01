"""v6 evaluation: A1800 against the hand-annotated reference (v5 vs v6) plus regression checks on the other three PDFs.
Usage (from floor-plan-detection): python -m src.evaluation.run_v6_evaluation [--v5 ../output_v5] [--v6 ../output_v6] [--out ../evaluation_v6]
Reads existing run folders (<plan>/rooms.json ...) - run them first with run.bat. Writes summary.json and full-page images (never cropped)."""
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from . import expected_ref as E
from .vector_proxies import proxies
from .vector_regression import compare

ROOT = Path(__file__).resolve().parents[2]
PDFS = ROOT.parent


def label(im, text):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (im.shape[1], 70), (255, 255, 255), -1)
    cv2.putText(im, text, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", default=str(PDFS / "output_v5"))
    ap.add_argument("--v6", default=str(PDFS / "output_v6"))
    ap.add_argument("--out", default=str(PDFS / "evaluation_v6"))
    ap.add_argument("--annotated", default=str(PDFS / "A1800 Ground Flr_expected.png"))
    a = ap.parse_args()
    v5, v6, out = Path(a.v5), Path(a.v6), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import yaml
    classes = yaml.safe_load(open(ROOT / "config" / "classes.yaml"))["classes"]
    S = {}

    # ---- A1800 against the annotated screenshot (a crop of the page at ~0.35x: scale hint = overlay px per screenshot px)
    ref = cv2.imread(a.annotated)
    S["annotated_reference"] = dict(file=Path(a.annotated).name, size=list(ref.shape[:2]))
    for tag, d in (("v5", v5 / "A1800_Ground_Flr"), ("v6", v6 / "A1800_Ground_Flr")):
        res = {}
        for off in (6, 10, 14):
            r = E.run(a.annotated, d, tag, save_dir=out / f"a1800_vs_annotation_{tag}" if off == 10 else None, networks=True, tols=(8, 16, 24), scale_hint=2.857, sep_offset_px=off)
            res[f"separated_at_{off}px"] = r["separation"]["separated"]
            res[f"separated_both_in_rooms_at_{off}px"] = r["separation"]["separated_both_in_rooms"]
            if off == 10:
                res.update(registration=r["registration"], recall=r["aggregate"], x_mark_pond_room_coverage=[x["max_room_polygon_coverage"] for x in r["x_marks"]],
                           per_line=[dict(id=p["id"], n=p["length_px"], recall_8px=p["recall@8px"], separated=q["separated"], separated_both_in_rooms=q["separated_both_in_rooms"])
                                     for p, q in zip(r["lines"], r["separation"]["per_line"])])
        S[f"a1800_{tag}"] = res
        S[f"a1800_{tag}"]["proxies"] = proxies(str(PDFS / "A1800 Ground Flr.pdf"), d, 200, 0.315, classes)

    # ---- regression checks
    S["student_v6_vs_v4_deliverable"] = compare(ROOT / "outputs" / "Student_Accomodation", v6 / "Student_Accomodation")
    fp = {}
    for n in ("floor_plan", "floor_plan2"):
        A = json.load(open(v5 / n / "rooms.json"))["rooms"]
        B = json.load(open(v6 / n / "rooms.json"))["rooms"]
        EA = json.load(open(v5 / n / "elements.json"))["elements"]
        EB = json.load(open(v6 / n / "elements.json"))["elements"]
        fp[n] = dict(rooms=[len(A), len(B)], polygons_identical=[r["polygon"] for r in A] == [r["polygon"] for r in B],
                     types_identical=[r["type"] for r in A] == [r["type"] for r in B], elements_identical=EA == EB)
    S["raster_plans_v6_vs_v5"] = fp
    r = E.run(PDFS / "expected.png", v6 / "floor_plan", "v6", save_dir=out / "floor_plan_expected_png_v6")
    S["floor_plan_expected_png_v6"] = dict(aggregate=r["aggregate"], x_marks=[x["max_room_polygon_coverage"] for x in r["x_marks"]])

    # ---- images (full page, uncropped)
    a5, a6 = cv2.imread(str(v5 / "A1800_Ground_Flr" / "overlay.png")), cv2.imread(str(v6 / "A1800_Ground_Flr" / "overlay.png"))
    cv2.imwrite(str(out / "A1800_v5_overlay_FULLPAGE.png"), a5)
    cv2.imwrite(str(out / "A1800_v6_overlay_FULLPAGE.png"), a6)
    sb = np.hstack([label(a5, "v5 (before)"), np.full((a5.shape[0], 12, 3), 0, np.uint8), label(a6, "v6 (after)")])
    cv2.imwrite(str(out / "A1800_v5_vs_v6_side_by_side_FULLPAGE.png"), cv2.resize(sb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    e5 = cv2.imread(str(out / "a1800_vs_annotation_v5" / "expected_vs_detected_fullpage.png"))
    e6 = cv2.imread(str(out / "a1800_vs_annotation_v6" / "expected_vs_detected_fullpage.png"))
    sb2 = np.hstack([label(e5, "v5: expected (red) vs detected (blue)"), np.full((e5.shape[0], 12, 3), 0, np.uint8), label(e6, "v6: expected (red) vs detected (blue)")])
    cv2.imwrite(str(out / "A1800_expected_vs_detected_v5_vs_v6_FULLPAGE.png"), cv2.resize(sb2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    for n, pdf in (("A1800_Ground_Flr", "A1800"), ("Student_Accomodation", "Student"), ("floor_plan", "floor_plan"), ("floor_plan2", "floor_plan2")):
        for suf in ("overlay.png", "masks.png", "rooms.json", "elements.json"):
            shutil.copyfile(v6 / n / f"{n}_{suf}", out / f"{n}_{suf}")
    json.dump(S, open(out / "summary.json", "w"), indent=1)
    a1, a2 = S["a1800_v5"], S["a1800_v6"]
    print(json.dumps({k: {"separated@10": v["separated_at_10px"], "both@10": v["separated_both_in_rooms_at_10px"], "recall@8": v["recall"]["pixel_recall@8px"], "mean_dist": v["recall"]["mean_dist_px"],
                          "rooms": v["proxies"]["n_rooms"], "labels_in_room": v["proxies"]["named_labels_inside_a_room"], "unknown_share": v["proxies"]["unknown_share"],
                          "rooms>150m2": v["proxies"]["large_rooms_over_150m2"], "boundary_on_wall": v["proxies"]["boundary_on_wall"]} for k, v in (("v5", a1), ("v6", a2))}, indent=1))
    print("student:", {k: S["student_v6_vs_v4_deliverable"][k] for k in ("rooms_new", "same_type", "min_iou", "max_vertex_displacement_pt")})
    print("raster:", fp)
    print("expected.png floor_plan v6:", {k: S["floor_plan_expected_png_v6"]["aggregate"][k] for k in ("mean_dist_px", "pixel_recall@8px", "lines_with_recall>=0.8@8px")})


if __name__ == "__main__":
    main()
