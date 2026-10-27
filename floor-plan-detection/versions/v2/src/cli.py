"""Usage: python -m src.cli <floor_plan.pdf> [--out outputs] [--no-ocr]"""
import argparse
from pathlib import Path
import yaml
from .pipeline import run_vector

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Detect rooms and architectural elements in a vector floor-plan PDF")
    ap.add_argument("pdf")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--config", default=str(ROOT / "config" / "inference.yaml"))
    ap.add_argument("--classes", default=str(ROOT / "config" / "classes.yaml"))
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-label-split", action="store_true", help="disable label-guided watershed split (ablation)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    classes = yaml.safe_load(open(a.classes))["classes"]
    if a.no_ocr:
        cfg["ocr"]["enabled"] = False
    if a.no_label_split:
        cfg.setdefault("instances", {})["split_by_labels"] = False
    run_vector(a.pdf, cfg, classes, a.out)


if __name__ == "__main__":
    main()
