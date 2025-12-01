import os, pickle, numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
import json

with open("model_metadata.json") as f:
    md = json.load(f)
seq_len = md["sequence_length"]; feat = md["feature_dim"]
model = load_model(md["model_file"])
with open("labels.pkl","rb") as f: labels = pickle.load(f)
labels_inv = {v:k for k,v in labels.items()}
scaler = pickle.load(open("scaler_lstm.pkl","rb"))

# collect files
test_folder = "gesture_data"
files=[]
for g in labels.keys():
    gpath=os.path.join(test_folder,g)
    for f in os.listdir(gpath):
        if f.endswith(".npy"): files.append((g,os.path.join(gpath,f)))

y_true=[]; y_pred=[]
for g,fpath in files:
    seq=np.load(fpath).astype(np.float32)
    # pad/trim to seq_len, zero pad
    if seq.shape[0] < seq_len:
        pad = np.zeros((seq_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)
    else:
        seq = seq[:seq_len]
    if seq.shape[1] != feat:
        fixed = np.zeros((seq_len,feat), dtype=np.float32); fixed[:,:min(seq.shape[1],feat)] = seq[:,:min(seq.shape[1],feat)]; seq=fixed
    seq_scaled = scaler.transform(seq.reshape(-1, seq.shape[-1])).reshape(1, seq_len, feat)
    pred = model.predict(seq_scaled, verbose=0)
    p = int(np.argmax(pred))
    y_true.append(labels[g])
    y_pred.append(p)

print(classification_report(y_true, y_pred, target_names=list(labels.keys())))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
