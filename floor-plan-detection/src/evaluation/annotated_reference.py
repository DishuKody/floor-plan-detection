"""Evaluate a run against a hand-annotated screenshot of a LARGE sheet (red lines drawn in Paint over a screenshot that is a crop of the page).
Usage: python -m src.evaluation.annotated_reference <annotated.png> <outputs_dir> <save_dir> [--scale-hint 2.857] [--tag v6]

Reports (all against the red lines; the red lines are sparse, so only recall-type measures are defined):
  distance recall    share of line pixels within 8 / 16 / 24 overlay px of ANY detected room boundary (weak: a dense polygon mesh scores well)
  separation         share of line pixels whose two sides (+-10 px) lie in DIFFERENT detected rooms - what an annotated room boundary means
  x marks            'not a shape' crosses: the largest room polygon coverage of the marked region (must be ~0)
The screenshot is registered to the page render by SIFT + RANSAC (affine); the reference file is never modified."""
import argparse
import json
from . import expected_ref as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference"); ap.add_argument("out_dir"); ap.add_argument("save_dir")
    ap.add_argument("--scale-hint", type=float, default=None, help="overlay px per screenshot px (the screenshot is a crop); default: assume the full page width")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--offset", type=int, default=10)
    a = ap.parse_args()
    r = E.run(a.reference, a.out_dir, a.tag, save_dir=a.save_dir, networks=True, tols=(8, 16, 24), scale_hint=a.scale_hint, sep_offset_px=a.offset)
    S = r["separation"]
    print(json.dumps(dict(registration=r["registration"], recall=r["aggregate"], separated=S["separated"], separated_both_in_rooms=S["separated_both_in_rooms"],
                          x_mark_coverage=[x["max_room_polygon_coverage"] for x in r["x_marks"]]), indent=1))
    for p, q in zip(r["lines"], S["per_line"]):
        print(f"  line {p['id']:2d} n {p['length_px']:5.0f} mean {p['mean_dist_px']:5.1f} recall@8 {p['recall@8px']:.2f}  separated {q['separated']:.2f} (both in rooms {q['separated_both_in_rooms']:.2f})")


if __name__ == "__main__":
    main()
