"""ONE COMMAND to finish the project on your machine:  trains Experiment 4 on real RAVDESS data, re-verifies and
re-selects the final model on VALIDATION Macro F1, rebuilds the PDF report and PowerPoint from the real numbers, and
runs the full release verification with a real test-actor audio clip.

    python finalize_project.py --name "Ronia Rosario K" --roll "YOUR_ID"

Prerequisite: RAVDESS in data\\RAVDESS (folders Actor_01..Actor_24) OR the downloaded Audio_Speech_Actors_01-24.zip placed in
data\\ (it is unzipped for you). Add --download to fetch it automatically if this PC has internet access.
Experiments 1-3 are never touched. Takes about 5-10 minutes on a normal CPU.
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="Ronia Rosario K"); ap.add_argument("--roll", default="")
ap.add_argument("--download", action="store_true"); ap.add_argument("--retrain", action="store_true", help="re-run Experiment 4 even if results exist")
ap.add_argument("--epochs", type=int, default=60)
a = ap.parse_args()


def run(cmd, must=True):
    print("\n>>", " ".join(cmd)); r = subprocess.run(cmd, cwd=ROOT)
    if must and r.returncode: sys.exit(f"\nSTOPPED: step failed ({' '.join(cmd)}). Fix the message above and run finalize_project.py again.")
    return r.returncode


py = sys.executable
if run([py, "check_env.py"], must=False):
    sys.exit("\nSTOPPED: environment not ready (needs Python 3.10-3.13 with TensorFlow). See README, then re-run.")

data = ROOT / "data" / "RAVDESS"
wavs = list(data.rglob("*.wav")) if data.exists() else []
if len(wavs) < 1440:
    z = next(iter((ROOT / "data").glob("Audio_Speech_Actors_01-24*.zip")), None) if (ROOT / "data").exists() else None
    if z:
        print(f"Unzipping {z.name} ..."); data.mkdir(parents=True, exist_ok=True); zipfile.ZipFile(z).extractall(data)
    elif a.download:
        pass                                              # run_experiment_04 --download below
    else:
        sys.exit("\nSTOPPED: RAVDESS not found. Download Audio_Speech_Actors_01-24.zip from https://zenodo.org/records/1188976, "
                 "put it in the 'data' folder (or unzip it into data\\RAVDESS), then re-run. Or add --download.")
    wavs = list(data.rglob("*.wav")) if data.exists() else []
if not a.download and len(wavs) != 1440:
    sys.exit(f"\nSTOPPED: expected 1440 speech clips, found {len(wavs)} in {data}. Use the 'Audio_Speech_Actors_01-24' (speech) archive.")

e4 = ROOT / "results" / "experiment_04" / "metrics.json"
if e4.exists() and not a.retrain:
    print("\nExperiment 4 results already exist - skipping training (use --retrain to redo).")
else:
    cmd = [py, "run_experiment_04.py", "--data", str(data), "--epochs", str(a.epochs)] + (["--download"] if a.download else []) + (["--overwrite"] if a.retrain else [])
    run(cmd)

run([py, "summarize_results.py"])
run([py, "build_report.py", "--name", a.name] + (["--roll", a.roll] if a.roll else []))
run([py, "build_ppt.py", "--name", a.name])
clip = sorted((data / "Actor_21").glob("*.wav"))[0] if (data / "Actor_21").exists() else sorted(data.rglob("Actor_21/*.wav"))[0]
rc = run([py, "verify_release.py", "--audio", str(clip)], must=False)

import json
S = json.loads((ROOT / "results" / "final" / "final_summary.json").read_text())
print("\n" + "=" * 70 + "\nFINAL RESULT SUMMARY")
for r in S["experiments"]:
    if r["status"] == "completed":
        print(f"  {r['label']:13s} val MacroF1 {r['val_macro_f1']:.4f} | test acc {r['test_accuracy']:.4f} | test MacroF1 {r['test_macro_f1']:.4f}")
print(f"  SVM baseline  test MacroF1 {S['svm_baseline']['test_macro_f1']:.4f}")
print(f"  SELECTED FINAL MODEL: {S['selected_final_model']['label']} (rule: highest validation Macro F1)")
print(f"\nReport:        {ROOT.parent / 'Speech_Emotion_Recognition_Report.pdf'}\nPresentation:  {ROOT.parent / 'Speech_Emotion_Recognition_Presentation.pptx'}")
print("RELEASE VERIFIED - ready to submit." if rc == 0 else "Verification reported problems - read the FAIL lines above.")
sys.exit(rc)
