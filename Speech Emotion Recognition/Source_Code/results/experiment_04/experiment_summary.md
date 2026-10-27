# Experiment 4 summary

- Dataset: RAVDESS (speech)
- Parameters: 79,080
- Best epoch (val Macro F1): 26
- Validation Macro F1: 0.4794
- Test accuracy: 0.5000
- Test Macro F1: 0.4783
- Test weighted F1: 0.4769
- Training time: 204.0 s
- Config: `{"model": "compact_cnn_bilstm_attention_exp4", "gaussian_noise": 0.08, "conv1": {"filters": 32, "kernel": 5, "dropout": 0.35}, "conv2": {"filters": 64, "kernel": 3, "dropout": 0.4}, "lstm": {"units": 48, "dropout": 0.35}, "attention_units": 32, "dense": {"units": 64, "dropout": 0.5}, "l2": 0.0002, "lr": 0.001, "batch_size": 32, "patience": 12, "aug_copies": 2, "seed": 42}`
