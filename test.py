# import os
# from config import Config

# counts = {}
# for gesture in os.listdir(Config.DATA_PATH):
#     path = os.path.join(Config.DATA_PATH, gesture)
#     if os.path.isdir(path):
#         counts[gesture] = len([f for f in os.listdir(path) if f.endswith('.npy')])
# print("Sample counts per gesture:")
# for g, c in counts.items():
#     print(f"{g:12s} → {c}")
# import shutil, random, os
# src = os.path.join(Config.DATA_PATH, "vishruth")
# files = [os.path.join(src, f) for f in os.listdir(src) if f.endswith('.npy')]
# for i in range(10):  # duplicate to balance
#     shutil.copy(random.choice(files), os.path.join(src, f"dup_{i}.npy"))
# import numpy as np, os

# base = "enhanced_gesture_data"
# for gesture in os.listdir(base):
#     gesture_path = os.path.join(base, gesture)
#     if os.path.isdir(gesture_path):
#         for f in os.listdir(gesture_path):
#             if f.endswith(".npy"):
#                 arr = np.load(os.path.join(gesture_path, f))
#                 if arr.shape != (30, 774) and arr.shape != (30, 2322):
#                     print(f"⚠️ Invalid shape {arr.shape} in {gesture}/{f}")
# import os
# import numpy as np

# # ✅ Path to your gesture data
# base_path = "enhanced_gesture_data"

# # ✅ Valid shapes your system supports
# valid_shapes = {(30, 774), (30, 2322)}

# print(f"🔍 Scanning '{base_path}' for corrupted .npy files...\n")
# corrupted_files = []

# for gesture in os.listdir(base_path):
#     gesture_dir = os.path.join(base_path, gesture)
#     if not os.path.isdir(gesture_dir):
#         continue

#     for file in os.listdir(gesture_dir):
#         if file.endswith(".npy"):
#             fpath = os.path.join(gesture_dir, file)
#             try:
#                 arr = np.load(fpath)
#                 if arr.shape not in valid_shapes:
#                     print(f"⚠️ Shape mismatch {arr.shape} → {gesture}/{file}")
#                     corrupted_files.append(fpath)
#             except Exception as e:
#                 print(f"❌ Corrupted or unreadable: {gesture}/{file} ({e})")
#                 corrupted_files.append(fpath)

# print("\n------------------------------------------")
# if corrupted_files:
#     print(f"⚠️ Found {len(corrupted_files)} corrupted or invalid files.")
#     choice = input("🗑️ Do you want to delete them? (y/n): ").strip().lower()
#     if choice == 'y':
#         for f in corrupted_files:
#             os.remove(f)
#             print(f"🗑️ Deleted: {f}")
#     else:
#         print("❎ Skipped deletion.")
# else:
#     print("✅ All .npy files are valid and ready for training!")
# import numpy as np, os
# base = "enhanced_gesture_data"  # adjust if different
# for g in os.listdir(base):
#     p = os.path.join(base, g)
#     for f in os.listdir(p):
#         if f.endswith(".npy"):
#             arr = np.load(os.path.join(p, f))
#             print(f"{g}/{f} → {arr.shape}")
#             break
# import numpy as np
# import os

# folder = "gesture_data/please"   # example: "gesture_data/Hello"

# for f in sorted(os.listdir(folder))[:3]:
#     arr = np.load(os.path.join(folder, f))
#     print(f, arr.shape)
# full_eval.py
# import os, json, pickle, numpy as np
# from sklearn.metrics import confusion_matrix, classification_report
# from tensorflow.keras.models import load_model
# from config import Config

# # Load metadata
# meta = json.load(open("model_metadata.json"))
# SEQ = meta["sequence_length"]
# FEAT = meta["feature_dim"]
# MODEL_FILE = meta["model_file"]

# print("Using model:", MODEL_FILE, "seq_len:", SEQ, "feat_dim:", FEAT)

# # Load model/scaler/labels
# model = load_model(MODEL_FILE)
# scaler = pickle.load(open("scaler_lstm.pkl","rb"))
# labels = pickle.load(open("labels.pkl","rb"))  # or Config.LABELS_PATH if that name is used
# labels_inv = {v:k for k,v in labels.items()}

# # prepare function (must match training preprocessing)
# def prepare(seq):
#     # pad/trim time
#     if seq.shape[0] < SEQ:
#         pad = np.tile(seq[-1:], (SEQ - seq.shape[0], 1))
#         seq = np.concatenate([seq, pad], axis=0)
#     elif seq.shape[0] > SEQ:
#         seq = seq[:SEQ]
#     # fix feature dim
#     if seq.shape[1] != FEAT:
#         fixed = np.zeros((SEQ, FEAT), dtype=np.float32)
#         fixed[:, :min(seq.shape[1], FEAT)] = seq[:, :min(seq.shape[1], FEAT)]
#         seq = fixed
#     # per-feature z-score (same as train)
#     seq = (seq - np.mean(seq, axis=0, keepdims=True)) / (np.std(seq, axis=0, keepdims=True) + 1e-6)
#     # scaler expects 2D rows=(time frames), cols=features
#     flat = seq.reshape(-1, FEAT)
#     flat_scaled = scaler.transform(flat)
#     seq_scaled = flat_scaled.reshape(SEQ, FEAT)
#     return seq_scaled.reshape(1, SEQ, FEAT)

# # collect dataset
# X, y_true = [], []
# classes = sorted([d for d in os.listdir(Config.DATA_PATH) if os.path.isdir(os.path.join(Config.DATA_PATH, d)) and not d.startswith("_")])
# label_to_idx = {c:i for i,c in enumerate(classes)}
# print("Classes:", classes)

# for c in classes:
#     cpath = os.path.join(Config.DATA_PATH, c)
#     for f in os.listdir(cpath):
#         if f.endswith(".npy"):
#             seq = np.load(os.path.join(cpath, f))
#             X.append(prepare(seq)[0])
#             y_true.append(label_to_idx[c])

# X = np.array(X)
# y_true = np.array(y_true)
# print("Samples:", len(X))

# # Predict in batches
# y_pred = []
# probs = model.predict(X, verbose=0)
# y_pred = probs.argmax(axis=1)

# # report
# print("\nClassification Report:\n")
# print(classification_report(y_true, y_pred, target_names=classes))
# cm = confusion_matrix(y_true, y_pred)
# print("\nConfusion matrix:\n", cm)

# # show top confusions
# import numpy as np
# cm2 = cm.copy().astype(float)
# np.fill_diagonal(cm2, 0)
# indices = np.dstack(np.unravel_index(np.argsort(cm2.ravel())[::-1], cm2.shape))[0]
# print("\nTop confusions (true -> pred):")
# for i,(t,p) in enumerate(indices[:10]):
#     if cm2[t,p] > 0:
#         print(f"  {classes[t]:<12} -> {classes[p]:<12} : {int(cm[t,p])}")

# # per-sample low-confidence list
# th = 0.65
# low = []
# for i,p in enumerate(probs):
#     if p.max() < th:
#         low.append((classes[y_true[i]], classes[y_pred[i]], float(p.max())))
# if low:
#     print(f"\nLow-confidence predictions (<{th}): {len(low)}")
#     for a,b,c in low[:20]:
#         print(f"  True:{a} Pred:{b} Conf:{c:.2f}")
# else:
#     print("\nNo low-confidence predictions found at threshold", th)
import os
import numpy as np
from config import Config

def check_hello():
    print("--- 🕵️ Checking 'Hello' Data ---")
    
    # Check if folder exists (Case Sensitive!)
    possible_names = ["Vishruth"]
    found_name = None
    for name in possible_names:
        if os.path.exists(os.path.join(Config.DATA_PATH, name)):
            found_name = name
            break
            
    if not found_name:
        print("❌ Folder 'Hello' (or hello) NOT found in gesture_data!")
        return

    g_path = os.path.join(Config.DATA_PATH, found_name)
    files = [f for f in os.listdir(g_path) if f.endswith('.npy')]
    print(f"📁 Found {len(files)} samples for '{found_name}'.")
    
    if len(files) < 15:
        print("⚠️ WARNING: Too few samples! You need at least 15-20.")
    
    # Check Motion (Right Hand Velocity Indices: 453-516)
    motions = []
    for f in files:
        try:
            data = np.load(os.path.join(g_path, f))
            # Check Right Hand Velocity
            vel = data[:, 453:516] 
            motion = np.mean(np.abs(vel))
            motions.append(motion)
        except: pass
        
    if not motions: return

    avg_motion = np.mean(motions)
    print(f"📊 Average Motion Score: {avg_motion:.5f}")
    
    if avg_motion < 0.0001:
        print("\n🔴 CRITICAL FAIL: Zero motion detected.")
        print("   - Did you use your LEFT hand? (System tracks RIGHT hand)")
        print("   - Was your hand out of frame?")
    elif avg_motion < 0.002:
         print("\n🟠 WARNING: Motion is very weak.")
         print("   - You are likely just holding your hand up.")
         print("   - The model ignores static poses.")
    else:
         print("\n✅ Motion looks good! The issue is likely confusion with 'Please'.")

if __name__ == "__main__":
    check_hello()