#
# Step 2: Model Training (RIDGE – FINAL ROBUST VERSION)
#
# ⚙️  Fast, regularized, works perfectly with real-time Mediapipe data
#

import os, numpy as np, pickle, logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import RidgeClassifier
from sklearn.utils import shuffle
from config import Config

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("--- Starting RidgeClassifier Training ---")

    if not os.path.exists(Config.DATA_PATH):
        logging.error(f"❌ Data path '{Config.DATA_PATH}' not found.")
        return

    gestures = sorted([d for d in os.listdir(Config.DATA_PATH) if os.path.isdir(os.path.join(Config.DATA_PATH, d))])
    if len(gestures) < 2:
        logging.error("Collect at least two gestures first.")
        return

    label_map = {label: idx for idx, label in enumerate(gestures)}
    X, y = [], []
    for g, idx in label_map.items():
        gpath = os.path.join(Config.DATA_PATH, g)
        for f in os.listdir(gpath):
            if f.endswith(".npy"):
                seq = np.load(os.path.join(gpath, f)).flatten()
                X.append(seq)
                y.append(g)
    X, y = np.array(X, np.float32), np.array(y)
    X, y = shuffle(X, y, random_state=42)
    # Normalize feature values to [0, 1]
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)


    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    Xtr, Xte = scaler.fit_transform(Xtr), scaler.transform(Xte)

    logging.info("🧠 Training RidgeClassifier...")
    model = RidgeClassifier(alpha=1.0)
    model.fit(Xtr, ytr)

    ypred = model.predict(Xte)
    acc = accuracy_score(yte, ypred)
    logging.info(f"✅ Validation Accuracy: {acc*100:.2f}%")
    print(classification_report(yte, ypred, target_names=gestures, zero_division=0))
    print(confusion_matrix(yte, ypred))

    with open(Config.MODEL_PATH, "wb") as f:
        pickle.dump({"model": model,
                     "label_map": {i: l for l, i in label_map.items()},
                     "scaler": scaler}, f)
    with open(Config.LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)

    logging.info("💾 Model & labels saved.")
    logging.info("--- TRAINING COMPLETE ---")

if __name__ == "__main__":
    main()
