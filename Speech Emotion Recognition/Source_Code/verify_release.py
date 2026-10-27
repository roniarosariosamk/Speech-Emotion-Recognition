"""One-command release verification.  Run inside the project .venv (Python 3.10-3.13, TensorFlow installed):

    python verify_release.py                       # uses a generated synthetic clip for the inference smoke test
    python verify_release.py --audio path\\to\\clip.wav   # recommended: a real RAVDESS clip

PASS requires: supported Python, all packages import, pytest finishes with ZERO skips and zero failures,
metrics verification passes, and the final model + its matching normaliser produce 8 finite probabilities summing to 1.
Writes results/final/verification_log.txt. Exit code 0 only if everything passed.
"""
import argparse
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--audio"); ap.add_argument("--experiment", default="experiment_03"); a = ap.parse_args()
rows, log = [], []


def step(name, ok, detail=""):
    rows.append((name, ok, detail)); print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True); out = r.stdout + r.stderr; log.append(f"$ {' '.join(cmd)}\n{out}"); return r.returncode, out


v = sys.version_info
step("Python version supported by TensorFlow 2.21 (3.10-3.13)", (3, 10) <= (v.major, v.minor) <= (3, 13), f"Python {platform.python_version()} on {platform.system()}")
rc, out = run([sys.executable, "check_env.py"]); step("check_env.py (all packages import)", rc == 0)
rc, out = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs"])
m = re.search(r"(\d+) passed", out); passed = int(m.group(1)) if m else 0
step("pytest: zero failures, zero skips", rc == 0 and "skipped" not in out and "error" not in out.lower().split("passed")[-1], f"{passed} passed")
rc, out = run([sys.executable, "summarize_results.py"]); step("summarize_results.py (metrics re-verified, model selected)", rc == 0, (re.search(r"\[final\] selected.*", out) or [""])[0] if rc == 0 else out[-200:])
try:
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import predict
    from ser.config import LABELS
    if a.audio:
        clip, kind = Path(a.audio), "REAL audio (functional check)"
    else:
        from ser import data as D
        clip = next(Path(D.make_synthetic_dataset(tempfile.mkdtemp(), n_actors=1, per_emotion=1)).rglob("*.wav")); kind = "SYNTHETIC audio (technical smoke test only)"
    p = predict.predict_proba(clip, a.experiment)
    ok = p.shape == (8,) and bool(np.all(np.isfinite(p))) and abs(float(p.sum()) - 1) < 1e-3
    step(f"predict.py: {a.experiment} model + normaliser -> 8 finite probabilities summing to 1", ok, f"{kind}; sum={p.sum():.6f}; top={LABELS[int(p.argmax())]}")
except Exception as e:  # noqa: BLE001
    step("predict.py inference smoke test", False, f"{e.__class__.__name__}: {e}")
ok_all = all(r[1] for r in rows)
(ROOT / "results" / "final").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "final" / "verification_log.txt").write_text(
    f"Python {platform.python_version()} | {platform.platform()}\n" + "\n".join(f"{'PASS' if o else 'FAIL'}  {n}  {d}" for n, o, d in rows) + "\n\n" + "\n".join(log))
print("\nRELEASE VERIFIED" if ok_all else "\nNOT VERIFIED - fix the FAIL items above"); sys.exit(0 if ok_all else 1)
