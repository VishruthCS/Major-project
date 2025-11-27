# test_model.py (FINAL FIXED VERSION)
import os
import json
import pickle
import random
import numpy as np
from tensorflow.keras.models import load_model
from config import Config

# ----- Load metadata -----
meta = json.load(open("model_metadata.json"))
seq_len_expected = meta["sequence_length"]
feat_dim_expected = meta["feature_dim"]
model_file = meta["model_file"]

print("Metadata:", meta)

# ----- Load model -----
model = load_model(model_file)

# ----- Load scaler + labels -----
scaler = pickle.load(open("scaler_lstm.pkl", "rb"))
labels = pickle.load(open("labels.pkl", "rb"))
labels_inv = {v: k for k, v in labels.items()}

# ----- Prepare sequence -----
def prepare_sequence(seq):
    # Pad/trim time dimension
    if seq.shape[0] < seq_len_expected:
        pad = np.tile(seq[-1:], (seq_len_expected - seq.shape[0], 1))
        seq = np.concatenate([seq, pad], axis=0)
    elif seq.shape[0] > seq_len_expected:
        seq = seq[:seq_len_expected]

    # Pad/trim feature dimension
    if seq.shape[1] != feat_dim_expected:
        fixed = np.zeros((seq_len_expected, feat_dim_expected), np.float32)
        fixed[:, :min(feat_dim_expected, seq.shape[1])] = seq[:, :min(feat_dim_expected, seq.shape[1])]
        seq = fixed

    # ----- FIX #1: Apply SAME Z-SCORE NORMALIZATION as training -----
    seq = (seq - np.mean(seq, axis=0, keepdims=True)) / (np.std(seq, axis=0, keepdims=True) + 1e-6)

    # Flatten for scaler
    seq_flat = seq.reshape(-1, feat_dim_expected)

    # Apply StandardScaler
    seq_scaled = scaler.transform(seq_flat).reshape(1, seq_len_expected, feat_dim_expected)

    return seq_scaled

# ----- Collect samples -----
all_files = []
for g in labels.keys():
    folder = os.path.join(Config.DATA_PATH, g)
    if not os.path.isdir(folder): continue
    for f in os.listdir(folder):
        if f.endswith(".npy"):
            all_files.append((g, os.path.join(folder, f)))

# ----- Predict random samples -----
samples = random.sample(all_files, min(5, len(all_files)))
for true_g, path in samples:
    seq = np.load(path)
    inp = prepare_sequence(seq)
    pred = model.predict(inp, verbose=0)[0]
    pred_idx = np.argmax(pred)
    pred_label = labels_inv[pred_idx]
    conf = float(np.max(pred))
    print(f"True: {true_g:<12} → Pred: {pred_label:<12} ({conf:.2f})")

print("✔ Testing completed.")
