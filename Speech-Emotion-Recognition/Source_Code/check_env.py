"""Check that the active Python environment can run the project.   python check_env.py"""
import importlib
import sys

REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "librosa", "soundfile", "matplotlib", "tensorflow", "keras"]
ok = True
v = sys.version_info
print(f"Python {v.major}.{v.minor}.{v.micro}")
if not (3, 10) <= (v.major, v.minor) <= (3, 13):
    ok = False
    print("  [!] TensorFlow 2.21 supports Python 3.10-3.13 only. Create the .venv with Python 3.13 "
          "(py -3.13 -m venv .venv) and reinstall: pip install -r requirements.txt")
for m in REQUIRED:
    try:
        mod = importlib.import_module(m)
        print(f"  [ok] {m:12s} {getattr(mod, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  [X]  {m:12s} not importable: {e.__class__.__name__}: {e}")
print("\nEnvironment OK" if ok else "\nEnvironment NOT ready - fix the items above, then run: python -m pytest -q")
sys.exit(0 if ok else 1)
