# test_model.py
import os, json, pickle, numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from config import Config

meta = json.load(open("model_metadata.json"))
SEQ = meta["sequence_length"]
FEAT = meta["feature_dim"]
MODEL_FILE = meta["model_file"]

print("✅ Model metadata:")
print("  expects sequence length", SEQ, "feature dim", FEAT)

model = load_model(MODEL_FILE)
scaler = pickle.load(open("scaler_lstm.pkl","rb"))
labels = pickle.load(open(Config.LABELS_PATH,"rb"))
labels_inv = {v:k for k,v in labels.items()}

def prepare(seq):
    seq = seq.astype('float32')
    if seq.shape[0] < SEQ:
        pad = np.tile(seq[-1:], (SEQ - seq.shape[0], 1))
        seq = np.concatenate([seq, pad], axis=0)
    elif seq.shape[0] > SEQ:
        seq = seq[:SEQ]
    if seq.shape[1] != FEAT:
        fixed = np.zeros((SEQ, FEAT), dtype=np.float32)
        trim = min(seq.shape[1], FEAT)
        fixed[:, :trim] = seq[:, :trim]
        seq = fixed
    # per-feature z-score across time only (as earlier design uses dataset-wide zscore in train)
    seq = (seq - np.mean(seq, axis=0, keepdims=True)) / (np.std(seq, axis=0, keepdims=True) + 1e-6)
    flat = seq.reshape(-1, FEAT)
    flat_scaled = scaler.transform(flat)
    seq_scaled = flat_scaled.reshape(SEQ, FEAT)
    return seq_scaled.reshape(1, SEQ, FEAT)

# collect all test files
X, y_true = [], []
classes = sorted([d for d in os.listdir(Config.DATA_PATH) if os.path.isdir(os.path.join(Config.DATA_PATH,d)) and not d.startswith("_")])
label_to_idx = {c:i for i,c in enumerate(classes)}

for c in classes:
    cpath = os.path.join(Config.DATA_PATH, c)
    for f in os.listdir(cpath):
        if f.endswith(".npy"):
            seq = np.load(os.path.join(cpath, f))
            X.append(prepare(seq)[0])
            y_true.append(label_to_idx[c])

X = np.array(X)
y_true = np.array(y_true)
print("Samples:", len(X), "Classes:", classes)

probs = model.predict(X, verbose=0)
y_pred = probs.argmax(axis=1)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=classes))
print("\nConfusion matrix:\n", confusion_matrix(y_true, y_pred))
