import numpy as np, random, pickle ,os
from tensorflow.keras.models import load_model
from config import Config

model = load_model("gesture_lstm_model.h5")
with open(Config.LABELS_PATH, "rb") as f:
    labels = pickle.load(f)
labels_inv = {v:k for k,v in labels.items()}

# Load 10 random test sequences
test_folder = "enhanced_gesture_data"
for g in random.sample(list(labels.keys()), 3):
    fpath = random.choice([os.path.join(test_folder, g, f) for f in os.listdir(os.path.join(test_folder, g)) if f.endswith(".npy")])
    seq = np.load(fpath)
    seq = seq.reshape(1, *seq.shape)
    pred = model.predict(seq)
    print(f"True: {g:<10} → Pred: {labels_inv[np.argmax(pred)]:<10} ({np.max(pred):.2f})")
