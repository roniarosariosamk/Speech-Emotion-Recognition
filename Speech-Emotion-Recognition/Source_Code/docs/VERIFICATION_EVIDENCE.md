# Verification evidence (release build, 26 Oct 2026)

Produced by `python verify_release.py` in a full environment: **Python 3.12.3 (Linux), TensorFlow 2.21.0, Keras 3.15.1, librosa 1.0.0, scikit-learn 1.8.0**. Full log: `results/final/verification_log.txt`.

| Step | Result |
|---|---|
| Python within TensorFlow 2.21 support (3.10-3.13) | PASS |
| `check_env.py` (all packages import) | PASS |
| `pytest`: zero failures, zero skips | PASS - 25 passed |
| `summarize_results.py`: metrics recomputed from confusion matrices, Experiment 3 selected | PASS - val Macro F1 0.5161, test Macro F1 0.5307 |
| `predict.py`: Experiment 3 model + matching `norm.npz` -> 8 finite probabilities, sum 1.000000 | PASS - **synthetic clip, technical smoke test only** |

Also executed: fresh extraction of the submission zip, `compileall`, `pytest` and `summarize_results.py` from that extraction (25 passed); a clean virtual environment **without** TensorFlow (11 passed, 10 skipped by design, `verify_release.py` correctly reports NOT VERIFIED).

## Honest limits of this evidence
- Run on Python 3.12 / Linux, not Python 3.13 / Windows. TensorFlow 2.21 supports both; re-run `verify_release.py` in your `.venv` to confirm on your machine.
- RAVDESS is not bundled (size). The final build ran `finalize_project.py`, whose verification used a REAL test-actor clip (Actor 21): 8 finite probabilities summing to 1.000000 (functional check, not an accuracy test).
- Metric checks prove the saved results are internally consistent; they do not reproduce the original training.
- Experiment 4 is a single training run (seed 42); its lower scores are measured, but run-to-run variation was not quantified.
