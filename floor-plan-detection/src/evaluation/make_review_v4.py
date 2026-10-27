"""Generates review_v4.md from evaluation_v4/summary.json (numbers are read from the file, never typed).
Usage: python -m src.evaluation.make_review_v4"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    S = json.load(open(ROOT / "evaluation_v4" / "summary.json"))
    e = S["expected_png"]
    a3, a4, reg = e["v3"]["aggregate"], e["v4"]["aggregate"], e["v4"]["registration"]

    def verdict(p3, p4):
        d = p4["recall@8px"] - p3["recall@8px"]
        return "better" if d > 0.05 else ("WORSE" if d < -0.05 else "same")
    rows = "\n".join(f"| {p4['id']} | {tuple(round(v) for v in p4['centre_overlay_px'])} | {p4['length_px']:.0f} | {p3['mean_dist_px']} | {p4['mean_dist_px']} | "
                     f"{p3['recall@8px']} | {p4['recall@8px']} | {verdict(p3, p4)} |" for p3, p4 in zip(e["v3"]["lines"], e["v4"]["lines"]))
    rb = S["raster_boundary_metrics"]
    rbrows = "\n".join(f"| {n} | {v} | {rb[n][v]['on_wall']} | {rb[n][v]['on_closure']} | {rb[n][v]['in_free_space']} | {rb[n][v]['on_text']} | {rb[n][v]['n_holes']} | "
                       f"{rb[n][v]['vertices_per_room']} | {rb[n][v]['mean_dist_to_nearest_wall_or_closure_px']} |" for n in rb for v in ("v3", "v4"))
    vr = S["vector_regression_student_v3_vs_v4"]
    worse = [p4["id"] for p3, p4 in zip(e["v3"]["lines"], e["v4"]["lines"]) if verdict(p3, p4) == "WORSE"]
    tpl = (ROOT / "src" / "evaluation" / "review_v4_template.md").read_text(encoding="utf-8")
    out = tpl.format(
        n_inl=reg["n_inliers"], rms=reg["rms_px"], sc=reg["scale_x"],
        m3=a3["mean_dist_px"], m4=a4["mean_dist_px"], md3=a3["median_dist_px"], md4=a4["median_dist_px"], p903=a3["p90_dist_px"], p904=a4["p90_dist_px"],
        r43=a3["pixel_recall@4px"], r83=a3["pixel_recall@8px"], r123=a3["pixel_recall@12px"], r44=a4["pixel_recall@4px"], r84=a4["pixel_recall@8px"], r124=a4["pixel_recall@12px"],
        l3=a3["lines_with_recall>=0.8@8px"], l4=a4["lines_with_recall>=0.8@8px"],
        x30=e["v3"]["x_marks"][0]["max_room_polygon_coverage"], x31=e["v3"]["x_marks"][1]["max_room_polygon_coverage"],
        rows=rows, worse=", ".join(str(i) for i in worse), rbrows=rbrows,
        vr_old=vr["rooms_old"], vr_new=vr["rooms_new"], vr_type=vr["same_type"], vr_iou=vr["min_iou"], vr_disp=vr["max_vertex_displacement_pt"], vr_el=vr["elements_new"],
        t1=S["floor_plan"]["runtime_s"], t2=S["floor_plan2"]["runtime_s"], t3=S["student_accomodation"]["runtime_s"],
        ram1=S["floor_plan"]["peak_ram_mb"], ram2=S["floor_plan2"]["peak_ram_mb"], ram3=S["student_accomodation"]["peak_ram_mb"])
    (ROOT / "review_v4.md").write_text(out, encoding="utf-8")
    print("review_v4.md written,", len(out), "chars")


if __name__ == "__main__":
    main()
