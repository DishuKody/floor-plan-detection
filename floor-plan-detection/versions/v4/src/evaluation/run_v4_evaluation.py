"""One-command reproduction of the v3 -> v4 evaluation.

    python -m src.evaluation.run_v4_evaluation [--out evaluation_v4] [--expected ../expected.png] [--device auto|cpu|cuda]

Runs the current pipeline on the three PDFs, then measures:
  * floor_plan.pdf vs expected.png  (registration, distance of the user's red lines to detected boundaries, X-mark check) for v3 and v4
  * floor_plan.pdf / floor_plan2.pdf reference-free boundary metrics for v3 and v4
  * Student Accomodation.pdf vector-mode regression against the preserved v3 outputs
  * runtime per stage, device, peak RAM
and writes full-page (never cropped) comparison images. The reference image is only read, never modified."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PDFS = ROOT.parent
V3 = ROOT / "versions" / "v3"


def run_pdf(pdf, out, device):
    from ..pipeline import run_vector
    cfg = yaml.safe_load(open(ROOT / "config" / "inference.yaml"))
    cfg.setdefault("compute", {})["device"] = device
    classes = yaml.safe_load(open(ROOT / "config" / "classes.yaml"))["classes"]
    t = time.time()
    run_vector(str(pdf), cfg, classes, out, log=lambda *a: None)
    st = json.load(open(Path(out) / "run_stats.json"))
    st["wall_s_including_ocr_model_load"] = round(time.time() - t, 2)
    return st


def side_by_side(a_png, b_png, out_png, la, lb):
    a, b = cv2.imread(str(a_png)), cv2.imread(str(b_png))
    h = max(a.shape[0], b.shape[0])
    pad = lambda x: cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for im, t in ((a, la), (b, lb)):
        cv2.rectangle(im, (0, 0), (im.shape[1], 34), (255, 255, 255), -1)
        cv2.putText(im, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_png), np.hstack([pad(a), np.full((h, 10, 3), 0, np.uint8), pad(b)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "evaluation_v4"))
    ap.add_argument("--expected", default=str(PDFS / "expected.png"))
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from . import expected_ref as ER
    from .raster_boundary_metrics import metrics as rb_metrics
    from .vector_regression import compare as vec_compare
    S = {}
    runs = {}
    for name, pdf in (("floor_plan", "floor_plan.pdf"), ("floor_plan2", "floor_plan2.pdf"), ("student_accomodation", "Student Accomodation.pdf")):
        d = out / name
        if d.exists():
            shutil.rmtree(d)
        runs[name] = run_pdf(PDFS / pdf, d, a.device)
        S[name] = dict(mode=runs[name].get("mode", "vector"), runtime_s=runs[name]["runtime_s"], device=runs[name]["device"],
                       stage_times_s=runs[name].get("stage_times_s"), n_rooms=runs[name]["n_rooms"], by_type=runs[name]["by_type"],
                       n_elements=runs[name]["n_elements"], peak_ram_mb=runs[name].get("peak_ram_mb"), gpu_measured=runs[name].get("gpu_measured"))
        print(name, S[name]["mode"], S[name]["runtime_s"], "s", S[name]["n_rooms"], "rooms", flush=True)

    # --- expected.png (floor_plan.pdf): v3 vs v4 -----------------------------------------------------------------------------------
    exp = {}
    for tag, d in (("v3", V3 / "raster_output"), ("v4", out / "floor_plan")):
        r = ER.run(a.expected, d, tag, save_dir=out / "expected_comparison" / tag, register_overlay=V3 / "raster_output" / "overlay.png")
        exp[tag] = dict(registration=r["registration"], aggregate=r["aggregate"], x_marks=r["x_marks"], lines=r["lines"])
    S["expected_png"] = exp
    for p in ("expected_vs_detected_fullpage.png",):
        side_by_side(out / "expected_comparison" / "v3" / p, out / "expected_comparison" / "v4" / p, out / "expected_comparison" / "v3_vs_v4_expected_red_detected_blue_FULLPAGE.png", "v3: expected (red) vs detected (blue)", "v4: expected (red) vs detected (blue)")

    # --- reference-free boundary metrics and overlay side-by-sides ---------------------------------------------------------------
    S["raster_boundary_metrics"] = {}
    for name, v3dir in (("floor_plan", V3 / "raster_output"), ("floor_plan2", V3 / "raster_output2")):
        ref = out / name
        S["raster_boundary_metrics"][name] = dict(v3=rb_metrics(v3dir, ref), v4=rb_metrics(ref, ref))
        side_by_side(v3dir / "overlay.png", ref / "overlay.png", out / f"{name}_v3_vs_v4_overlay_FULLPAGE.png", "v3", "v4")

    # --- complex plan: no degradation --------------------------------------------------------------------------------------------
    S["vector_regression_student_v3_vs_v4"] = vec_compare(V3 / "outputs", out / "student_accomodation")
    json.dump(S, open(out / "summary.json", "w"), indent=1)
    write_md(S, out / "summary.md")
    print("wrote", out / "summary.md")


def write_md(S, path):
    e3, e4 = S["expected_png"]["v3"]["aggregate"], S["expected_png"]["v4"]["aggregate"]
    reg = S["expected_png"]["v4"]["registration"]
    L = ["# v3 -> v4 evaluation (generated by `python -m src.evaluation.run_v4_evaluation`)", "",
         "## 1. `expected.png` (floor_plan.pdf) - hand-drawn red lines = expected boundaries",
         f"Registration of the screenshot to the overlay: {reg['n_inliers']} inliers, RMS {reg['rms_px']} px, scale {reg['scale_x']}.", "",
         "| metric (13 expected lines, overlay px at 2.0 px/pt) | v3 | v4 |", "|---|---|---|"]
    for k, lab in (("mean_dist_px", "mean distance to nearest detected boundary"), ("median_dist_px", "median distance"), ("p90_dist_px", "90th percentile"),
                   ("pixel_recall@4px", "pixel recall @4px"), ("pixel_recall@8px", "pixel recall @8px"), ("pixel_recall@12px", "pixel recall @12px"),
                   ("lines_with_recall>=0.8@8px", "lines with >=80% recall @8px (of 13)")):
        L.append(f"| {lab} | {e3[k]} | {e4[k]} |")
    L += ["", "Per expected line (mean distance px / recall@8px), v3 -> v4:", "", "| line | centre (overlay px) | length px | v3 mean | v4 mean | v3 recall@8 | v4 recall@8 |", "|---|---|---|---|---|---|---|"]
    for p3, p4 in zip(S["expected_png"]["v3"]["lines"], S["expected_png"]["v4"]["lines"]):
        L.append(f"| {p4['id']} | {p4['centre_overlay_px']} | {p4['length_px']:.0f} | {p3['mean_dist_px']} | {p4['mean_dist_px']} | {p3['recall@8px']} | {p4['recall@8px']} |")
    L += ["", "X marks (title text / logo are NOT shapes): max room-polygon coverage of the marked regions  v3: "
          + ", ".join(str(x["max_room_polygon_coverage"]) for x in S["expected_png"]["v3"]["x_marks"]) + "  v4: "
          + ", ".join(str(x["max_room_polygon_coverage"]) for x in S["expected_png"]["v4"]["x_marks"]), "",
          "## 2. Reference-free boundary metrics (one neutral wall mask for both versions)", "",
          "| plan | version | boundary on wall | on wall-extension | in free space | on text | holes | vertices/room | mean dist to wall/closure px |", "|---|---|---|---|---|---|---|---|---|"]
    for n, d in S["raster_boundary_metrics"].items():
        for v in ("v3", "v4"):
            m = d[v]
            L.append(f"| {n} | {v} | {m['on_wall']} | {m['on_closure']} | {m['in_free_space']} | {m['on_text']} | {m['n_holes']} | {m['vertices_per_room']} | {m['mean_dist_to_nearest_wall_or_closure_px']} |")
    vr = S["vector_regression_student_v3_vs_v4"]
    L += ["", "## 3. Complex vector plan (Student Accomodation.pdf): v3 vs v4", "",
          f"rooms {vr['rooms_old']} -> {vr['rooms_new']}, same type {vr['same_type']}, min IoU {vr['min_iou']}, mean IoU {vr['mean_iou']}, max vertex displacement {vr['max_vertex_displacement_pt']} pt; elements {vr['elements_old']} -> {vr['elements_new']}", "",
          "## 4. Runtime", "", "| pdf | mode | device | runtime s | peak RAM MB |", "|---|---|---|---|---|"]
    for n in ("floor_plan", "floor_plan2", "student_accomodation"):
        s = S[n]
        L.append(f"| {n} | {s['mode']} | {s['device']} | {s['runtime_s']} | {s['peak_ram_mb']} |")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
