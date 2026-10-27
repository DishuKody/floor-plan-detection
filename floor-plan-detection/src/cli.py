"""Usage: python -m src.cli <floor_plan.pdf> [--out outputs] [--no-ocr]"""
import argparse
import re
import shutil
from pathlib import Path
import yaml
from .pipeline import run_vector

ROOT = Path(__file__).resolve().parent.parent


def deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def main():
    ap = argparse.ArgumentParser(description="Detect rooms and architectural elements in a vector floor-plan PDF")
    ap.add_argument("pdf")
    ap.add_argument("--out", default=None, help="output folder (default: outputs/<input-file-name>/)")
    ap.add_argument("--config", default=str(ROOT / "config" / "inference.yaml"))
    ap.add_argument("--classes", default=str(ROOT / "config" / "classes.yaml"))
    ap.add_argument("--profile", default=None, help="sheet profile yaml (config/profiles/*.yaml) merged over the config: printed scale, symbol exemplars, ...")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-gpu-sampling", action="store_true", help="do not poll nvidia-smi (cleaner timing)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], help="compute device (default: config compute.device, auto = CUDA if PyTorch sees it)")
    ap.add_argument("--no-label-split", action="store_true", help="disable label-guided watershed split (ablation)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.profile:
        cfg = deep_merge(cfg, yaml.safe_load(open(a.profile)))
    classes = yaml.safe_load(open(a.classes))["classes"]
    if a.device:
        cfg.setdefault("compute", {})["device"] = a.device
    if a.no_gpu_sampling:
        cfg.setdefault("compute", {})["sample_gpu"] = False
    if a.no_ocr:
        cfg["ocr"]["enabled"] = False
    if a.no_label_split:
        cfg.setdefault("instances", {})["split_by_labels"] = False
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(a.pdf).stem).strip("_") or "floorplan"
    out = Path(a.out) if a.out else ROOT / "outputs" / stem
    run_vector(a.pdf, cfg, classes, out)
    # named copies of the main deliverables: <input-name>_<what>.<ext>
    named = {"overlay.png": f"{stem}_overlay.png", "room_masks.png": f"{stem}_masks.png", "rooms.json": f"{stem}_rooms.json", "elements.json": f"{stem}_elements.json"}
    for src, dst in named.items():
        shutil.copyfile(out / src, out / dst)
    print("FINAL OUTPUT FILES in", out.resolve())
    for dst in named.values():
        print("  ", dst)


if __name__ == "__main__":
    main()
