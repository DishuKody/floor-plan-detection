"""GPU helpers: device selection with CPU fallback, GPU template matching (PyTorch) and an nvidia-smi sampler that MEASURES
utilisation / VRAM (nothing here assumes the GPU is used - callers report what the sampler saw)."""
import subprocess
import threading
import time
import numpy as np


def resolve_device(pref="auto"):
    """-> ('cuda'|'cpu', info dict). 'auto' uses CUDA only if PyTorch sees it; 'cuda' raises if unavailable; 'cpu' never imports torch."""
    if pref == "cpu":
        return "cpu", dict(reason="forced cpu")
    try:
        import torch
    except Exception as e:                       # torch not installed
        if pref == "cuda":
            raise RuntimeError(f"device=cuda requested but PyTorch is not importable: {e}")
        return "cpu", dict(reason=f"torch not importable ({type(e).__name__})")
    if not torch.cuda.is_available():
        if pref == "cuda":
            raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False")
        return "cpu", dict(reason="torch.cuda.is_available() is False")
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    return "cuda", dict(name=p.name, capability=f"{p.major}.{p.minor}", vram_total_mb=round(total / 2 ** 20), vram_free_mb=round(free / 2 ** 20),
                        torch=torch.__version__, cuda_runtime=torch.version.cuda)


class FftCorrelator:
    """Normalised cross-correlation (== cv2.TM_CCOEFF_NORMED, 'valid' size) of many templates against ONE image on the GPU.
    The image FFT and the window-sum integral images are computed once and reused for every template (variants/exemplars).
    Numerator via FFT (fp32); window sums / variance via float64 integral images (avoids catastrophic cancellation)."""

    def __init__(self, image, device="cuda"):
        import torch
        self.torch, self.dev = torch, torch.device(device)
        self.H, self.W = image.shape
        I = torch.from_numpy(np.ascontiguousarray(image, np.float32)).to(self.dev) / 255.0
        self.FI = torch.fft.rfft2(I)
        I64 = I.double()
        self.S1 = self._integral(I64)
        self.S2 = self._integral(I64 * I64)
        del I, I64

    @staticmethod
    def _integral(X):
        import torch.nn.functional as F
        return F.pad(X.cumsum(0).cumsum(1), (1, 0, 1, 0))

    def match(self, template):
        torch = self.torch
        h, w = template.shape
        H, W = self.H, self.W
        T = torch.from_numpy(np.ascontiguousarray(template, np.float32)).to(self.dev) / 255.0
        N = float(h * w)
        T0 = T - T.mean()
        tnorm = torch.sqrt((T0.double() ** 2).sum())
        Tp = torch.zeros((H, W), device=self.dev)
        Tp[:h, :w] = T0
        num = torch.fft.irfft2(self.FI * torch.fft.rfft2(Tp).conj(), s=(H, W))[:H - h + 1, :W - w + 1].double()
        box = lambda S: S[h:, w:][:H - h + 1, :W - w + 1] - S[:-h, w:][:H - h + 1, :W - w + 1] - S[h:, :-w][:H - h + 1, :W - w + 1] + S[:-h, :-w][:H - h + 1, :W - w + 1]
        s1, s2 = box(self.S1), box(self.S2)
        den = torch.sqrt(torch.clamp(s2 - s1 * s1 / N, min=0)) * tnorm
        R = torch.where(den > 1e-6, num / den, torch.zeros_like(num))
        return R.float().cpu().numpy()


def match_template_ccoeff_normed(image, template, device="cuda"):
    """One-shot convenience wrapper (builds a correlator for a single template)."""
    return FftCorrelator(image, device).match(template)


class GpuSampler:
    """Polls `nvidia-smi` in a thread: utilisation (%) and used VRAM (MiB) of GPU 0 for the duration of a `with` block."""

    def __init__(self, interval=0.2):
        self.interval, self.samples, self._stop = interval, [], threading.Event()

    def _read(self):
        try:
            o = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True, timeout=5)
            u, m = [float(x) for x in o.strip().split(",")[:2]]
            return u, m
        except Exception:
            return None

    def __enter__(self):
        r = self._read()
        self.baseline_mem = r[1] if r else None
        self._t0 = time.time()

        def loop():
            while not self._stop.is_set():
                r = self._read()
                if r:
                    self.samples.append(r)
                time.sleep(self.interval)
        self._th = threading.Thread(target=loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._th.join(timeout=3)

    def summary(self):
        if not self.samples:
            return dict(available=False)
        u = np.array([s[0] for s in self.samples]); m = np.array([s[1] for s in self.samples])
        return dict(available=True, n_samples=len(u), interval_s=self.interval, baseline_vram_used_mb=self.baseline_mem,
                    peak_vram_used_mb=float(m.max()), vram_delta_over_baseline_mb=float(m.max() - (self.baseline_mem or 0)),
                    peak_util_pct=float(u.max()), mean_util_pct=round(float(u.mean()), 1), samples_over_10pct=int((u > 10).sum()))


def peak_ram_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().peak_wset / 2 ** 20)
    except Exception:
        return None
