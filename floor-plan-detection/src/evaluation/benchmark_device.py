"""Usage: python -m src.evaluation.benchmark_device <pdf> <out_dir> [--rounds 5]
Interleaved CPU-vs-GPU benchmark (the laptop's speed drifts: battery / thermal / background load, so configurations are run
round-robin in rotating order and compared by median, with min-max shown).

Configurations (each run is a fresh subprocess = cold process, warm disk cache):
  cpu_only_env    .venv_cpuref (no PyTorch, CPU-only onnxruntime 1.31)   --device cpu     <- the fair CPU reference
  gpu_env_cpu     .venv (PyTorch cu128 + onnxruntime-gpu 1.23.2)         --device cpu     <- same env, GPU switched off
  gpu_env_cuda    .venv                                                  --device cuda    <- RTX 2050
GPU utilisation / VRAM come from one extra sampled run (nvidia-smi polling slows the process, so it is never used for timing)."""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = [("cpu_only_env", ROOT / ".venv_cpuref" / "Scripts" / "python.exe", "cpu"),
           ("gpu_env_cpu", ROOT / ".venv" / "Scripts" / "python.exe", "cpu"),
           ("gpu_env_cuda", ROOT / ".venv" / "Scripts" / "python.exe", "cuda")]


def run_once(py, pdf, out, device, sample=False, profile=None):
    t = time.time()
    cmd = [str(py), "-m", "src.cli", str(pdf), "--out", str(out), "--device", device] + ([] if sample else ["--no-gpu-sampling"]) + (["--profile", str(profile)] if profile else [])
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    wall = time.time() - t
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    st = json.load(open(Path(out) / "run_stats.json"))
    return dict(process_wall_s=round(wall, 2), pipeline_s=st["runtime_s"], stages=st["stage_times_s"], gpu=st.get("gpu_measured"),
                ocr_providers=st.get("ocr_providers"), peak_ram_mb=st.get("peak_ram_mb"), device=st["device"])


def same_results(a, b):
    ra, rb = json.load(open(Path(a) / "rooms.json"))["rooms"], json.load(open(Path(b) / "rooms.json"))["rooms"]
    ea, eb = json.load(open(Path(a) / "elements.json"))["elements"], json.load(open(Path(b) / "elements.json"))["elements"]
    oa, ob = json.load(open(Path(a) / "debug/ocr.json")), json.load(open(Path(b) / "debug/ocr.json"))
    ta, tb = sorted(t["text"] for t in oa), sorted(t["text"] for t in ob)
    cnt = lambda E, k: sum(e["type"] == k for e in E)
    return dict(n_rooms=(len(ra), len(rb)), room_types_identical=[r["type"] for r in ra] == [r["type"] for r in rb],
                room_polygons_identical=len(ra) == len(rb) and all(x["polygon_pt"] == y["polygon_pt"] for x, y in zip(ra, rb)),
                windows=(cnt(ea, "window"), cnt(eb, "window")), doors=(cnt(ea, "door"), cnt(eb, "door")),
                ocr_boxes=(len(oa), len(ob)), ocr_text_sets_identical=(ta == tb),
                ocr_texts_only_a=sorted(set(ta) - set(tb))[:15], ocr_texts_only_b=sorted(set(tb) - set(ta))[:15])


def stat(v):
    return dict(median=round(statistics.median(v), 2), min=round(min(v), 2), max=round(max(v), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf"); ap.add_argument("out"); ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--profile", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = {name: [] for name, *_ in CONFIGS}
    for r in range(a.rounds):
        order = CONFIGS[r % len(CONFIGS):] + CONFIGS[:r % len(CONFIGS)]          # rotate to avoid position bias
        for name, py, dev in order:
            res = run_once(py, a.pdf, out / f"{name}_r{r + 1}", dev, profile=a.profile)
            runs[name].append(res)
            print(f"round {r + 1} {name:13s} pipeline {res['pipeline_s']:5.1f}s  wall {res['process_wall_s']:5.1f}s", flush=True)
    sampled = run_once(ROOT / ".venv" / "Scripts" / "python.exe", a.pdf, out / "gpu_env_cuda_sampled", "cuda", sample=True, profile=a.profile)
    summ = {}
    for name, rs in runs.items():
        summ[name] = dict(pipeline_s=stat([x["pipeline_s"] for x in rs]), process_wall_s=stat([x["process_wall_s"] for x in rs]),
                          stage_median_s={k: round(statistics.median(x["stages"][k] for x in rs), 2) for k in rs[0]["stages"]},
                          peak_ram_mb=max(x["peak_ram_mb"] or 0 for x in rs), ocr_providers=sorted({p for x in rs[-1:] for v in (x["ocr_providers"] or {}).values() for p in v}))
    summ["gpu_env_cuda"]["gpu_measured_in_separate_sampled_run"] = sampled["gpu"]
    summ["gpu_env_cuda"]["sampled_run_pipeline_s"] = sampled["pipeline_s"]
    ref, g = summ["cpu_only_env"], summ["gpu_env_cuda"]
    summ["speedup_cuda_vs_cpu_only_env_median"] = dict(pipeline=round(ref["pipeline_s"]["median"] / g["pipeline_s"]["median"], 2),
                                                       process_wall=round(ref["process_wall_s"]["median"] / g["process_wall_s"]["median"], 2),
                                                       ocr_stage=round(ref["stage_median_s"]["ocr"] / g["stage_median_s"]["ocr"], 2))
    cmp = same_results(out / f"cpu_only_env_r{a.rounds}", out / f"gpu_env_cuda_r{a.rounds}")
    json.dump(dict(runs=runs, summary=summ, cpu_only_vs_cuda_results=cmp, sampled_run=sampled,
                   conditions=dict(note="laptop on battery / Balanced plan during benchmarking; timings drift, hence interleaving")), open(out / "gpu_benchmark.json", "w"), indent=1)
    print(json.dumps(summ, indent=1)); print(json.dumps(cmp, indent=1))


if __name__ == "__main__":
    main()
