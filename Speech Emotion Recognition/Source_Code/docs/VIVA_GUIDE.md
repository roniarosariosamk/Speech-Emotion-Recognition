# Viva preparation (beginner friendly)

**What does the project do?** It listens to a short speech clip and decides which of 8 emotions it expresses: neutral, calm, happy, sad, angry, fearful, disgust, surprised.

**Why MFCC?** Raw audio is thousands of numbers per second. MFCCs compress each tiny slice of sound into 40 numbers that describe the voice's timbre in a way similar to human hearing. They are a standard, proven speech feature.

**Why delta and delta-delta?** Emotion is in how the voice *changes*. Delta = speed of change, delta-delta = acceleration. They turn 40 features into 120 per frame and tell the model about movement.

**Why a CNN?** Small filters slide over time and detect short sound patterns (like edge detectors in images), and reduce 130 frames to 32, which makes the next stage cheaper and less prone to over-fitting.

**Why a BiLSTM?** Emotion unfolds over time. An LSTM remembers earlier sounds; "bi" means it reads the clip forwards and backwards, so each moment knows what came before and after.

**Why attention?** Not all moments are equally emotional. Attention learns a weight per frame so important moments (stressed syllables, pitch peaks) count more than pauses.

**Why Macro F1?** Accuracy can look good if the model is great on easy emotions and bad on others. Macro F1 averages the F1 of all 8 classes equally, so every emotion matters. F1 combines precision (are my predictions right?) and recall (did I find them all?).

**How was the data split?** By actor: actors 1-16 train, 17-20 validation (tuning, checkpoint choice), 21-24 test (final score only). No actor is in two groups, so the test measures performance on *new voices*.

**Why not a random split?** The same speaker would appear in train and test, and the model could recognise the speaker instead of the emotion, inflating the score.

**What prevents data leakage?** Normalisation statistics come from training data only; augmentation only on training clips; the SVM's settings are tuned on validation only; the final model is chosen by validation Macro F1.

**How does the SVM compare?** SVM (summary statistics of MFCC, 240 numbers): test Macro F1 0.4464. Deep model (Exp 3): 0.5307, i.e. +0.084. The sequence model uses time information the SVM throws away.

**What were the experiments?** Exp 1 wide model (0.4292) -> Exp 2 compact model with more regularisation (0.5231) -> Exp 3 same model, checkpoint chosen by validation Macro F1 (0.5307). Exp 4 (slightly more dropout) was trained once: val Macro F1 0.4794 and test Macro F1 0.4783, both lower than Exp 3, so Exp 3 stays final. Single seed, so this shows more dropout did not help in this run, not that it can never help.

**What are the weaknesses?** Sad recall 0.156 (confused with calm 12/32, neutral 8/32, disgust 6/32), fearful recall 0.250 (confused with angry, sad, happy), neutral over-predicted (precision 0.297). Possible reasons (not proven): similar low-energy speech, shared arousal, few training actors. Acted data only; four test actors; not a general emotion detector.

**Why choose Exp 3 and not the newest experiment?** Selection rule: highest validation Macro F1. Exp 3 also has the best test Macro F1. We never choose a model just because it is newer.

**If asked "is 53% good?"** With 8 classes chance is 12.5%; actor-independent SER on RAVDESS is hard and published numbers vary widely with protocol, so compare against the SVM baseline under the same protocol (+0.084 Macro F1) and be clear about uncertainty.
