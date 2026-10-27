# Syncing this release with GitHub

GitHub `main` (commit `dabfd26`, 25 Oct 2026) is the pre-audit version: it lacks `ser/final.py`, `ser/train_exp4.py`, `run_experiment_04.py`, `summarize_results.py`, `check_env.py`, `tests/test_project.py`, `results/final/`, `docs/`, and it contains the older `predict.py` (loads Experiment 1), `ser/model.py`, `ser/train.py`, `ser/evaluate.py`, plus an old `Final_Submission.zip` in the repository root.

This release is the canonical version. Sync with a normal commit (history is kept; nothing is force-pushed):

```powershell
git clone https://github.com/roniarosariosamk/Speech-Emotion-Recognition sync_tmp   # or use your existing clone
cd sync_tmp
git checkout -b final-audited-release
# copy the contents of this release over the clone (PowerShell; adjust the path to the unzipped release)
Copy-Item -Recurse -Force "<unzipped>\Speech Emotion Recognition\*" ".\Speech Emotion Recognition\"
git status                       # review every change
git add -A
git commit -m "Final audited release: Experiment 3 final model, Experiment 4 code, tests, docs, verified report"
git push -u origin final-audited-release
```
Open a pull request into `main` (or merge locally with `git checkout main; git merge final-audited-release; git push`).
Decide separately whether to remove the stale root `Final_Submission.zip` from the repository (`git rm Final_Submission.zip`); it is not needed.
Large files: model checkpoints are about 1-4 MB each and are fine for normal git; `data/RAVDESS/` is ignored.
