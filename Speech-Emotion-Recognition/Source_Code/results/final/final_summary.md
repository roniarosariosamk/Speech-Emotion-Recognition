# Final results summary

**Project:** Speech Emotion Recognition - MFCC + CNN-BiLSTM-Attention (RAVDESS)

**Dataset:** RAVDESS (speech) - 1440 clips, 24 actors, actor-independent split (train 1-16, val 17-20, test 21-24), seed 42

## Experiment comparison

| Experiment | Parameters | Epochs run (best) | Train time (s) | Val Macro F1 | Test accuracy | Test Macro F1 | Test weighted F1 |
|---|---|---|---|---|---|---|---|
| Experiment 1 | 341,064 | 17 (5) | 83 | 0.4484 | 0.4708 | 0.4292 | 0.4323 |
| Experiment 2 | 79,080 | 24 (12) | 147 | 0.4873 | 0.5375 | 0.5231 | 0.5343 |
| Experiment 3 **(selected)** | 79,080 | 24 (12) | 170 | 0.5161 | 0.5667 | 0.5307 | 0.5384 |
| Experiment 4 | 79,080 | 38 (26) | 204 | 0.4794 | 0.5000 | 0.4783 | 0.4769 |
| SVM baseline (RBF, C=100, gamma=0.001) | - | - | - | - | 0.4667 | 0.4464 | 0.4624 |

**Experiment 4 status:** completed

## Selected final model: Experiment 3

- Checkpoint: `models\experiment_03\ser_model_best.keras`  |  Normaliser: `models\experiment_03\norm.npz`
- Selection rule: highest validation Macro F1 among completed real-data experiments (test set not used for selection)
- Validation Macro F1 0.5161 | Test accuracy 0.5667 | Macro precision 0.5743 | Macro recall 0.5742 | Macro F1 0.5307 | Weighted F1 0.5384
- Macro F1 gain over SVM baseline: +0.0843

### Per-class test results

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| neutral | 0.2973 | 0.6875 | 0.4151 | 16 |
| calm | 0.5897 | 0.7188 | 0.6479 | 32 |
| happy | 0.4783 | 0.3438 | 0.4000 | 32 |
| sad | 0.4545 | 0.1562 | 0.2326 | 32 |
| angry | 0.5918 | 0.9062 | 0.7160 | 32 |
| fearful | 0.8000 | 0.2500 | 0.3810 | 32 |
| disgust | 0.6111 | 0.6875 | 0.6471 | 32 |
| surprised | 0.7714 | 0.8438 | 0.8060 | 32 |

### Most frequent test confusions (observed)

- sad -> calm: 12 clips (38% of sad)
- happy -> neutral: 8 clips (25% of happy)
- happy -> surprised: 8 clips (25% of happy)
- sad -> neutral: 8 clips (25% of sad)
- calm -> neutral: 7 clips (22% of calm)
- fearful -> angry: 7 clips (22% of fearful)

## Limitations

- RAVDESS is acted speech from 24 actors and two sentences; real-world, noisy or spontaneous speech will behave differently.
- The test set contains 240 clips from 4 actors, so every metric has substantial sampling variance; differences of a few F1 points between experiments are not statistically established.
- Experiment 3 was selected using validation Macro F1. However, test results were also examined when comparing Experiments 1-3, so repeated model development informed by the test results may make the reported test performance somewhat optimistic.
- Sad and fearful speech have low recall; neutral is over-predicted (low precision). Explanations (acoustic similarity, small data) are plausible but were not tested experimentally.
- Only MFCC-based features were used; no cross-corpus or noise-robustness evaluation was performed.
