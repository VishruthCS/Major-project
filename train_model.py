#
# train_lstm_model.py — Enhanced LSTM (Model 3, Final Polished)
#
# ✅ Sequence-aware deep LSTM
# ✅ Uses velocity + acceleration features (smart textbook)
# ✅ Dropout regularization & ReLU dense layer
# ✅ Per-feature normalization (prevents class collapse)
# ✅ Exports .h5 model for real-time recognition
#

import os
import numpy as np
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GroupShuffleSplit
from config import Config

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Early stopping ---
callbacks = [EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)]


def main():
    logging.info("--- Starting Enhanced LSTM Training ---")

    if not os.path.exists(Config.DATA_PATH):
        logging.error(f"❌ Data path '{Config.DATA_PATH}' not found.")
        return

    # --- Load all gesture sequences ---
    gestures = sorted([d for d in os.listdir(Config.DATA_PATH) if os.path.isdir(os.path.join(Config.DATA_PATH, d))])
    label_map = {gesture: idx for idx, gesture in enumerate(gestures)}
    features, labels = [], []

    logging.info("📂 Loading gesture sequences...")
    for gesture, idx in label_map.items():
        gesture_dir = os.path.join(Config.DATA_PATH, gesture)
        for file in os.listdir(gesture_dir):
            if file.endswith(".npy"):
                seq = np.load(os.path.join(gesture_dir, file))
                # ensure correct shape
                if seq.shape == (Config.SEQUENCE_LENGTH, 774):
                    features.append(seq)
                    labels.append(idx)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    logging.info(f"✅ Loaded {len(X)} sequences across {len(label_map)} gestures")

    if len(X) == 0:
        logging.error("❌ No valid .npy files found. Check your dataset.")
        return

    # --- Normalize features (critical for stability) ---
    # Per-feature z-score normalization across all samples
    X = (X - np.mean(X, axis=(0, 1), keepdims=True)) / (np.std(X, axis=(0, 1), keepdims=True) + 1e-6)

    # --- Flatten across time for scaling (extra normalization) ---
    nsamples, nframes, nfeatures = X.shape
    X_reshaped = X.reshape(nsamples * nframes, nfeatures)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(nsamples, nframes, nfeatures)

    # --- Train-test split ---
    groups = []
    for gesture, idx in label_map.items():
        gesture_dir = os.path.join(Config.DATA_PATH, gesture)
        for file in os.listdir(gesture_dir):
            if file.endswith(".npy"):
                base_name = file.replace("dup_", "").split(".")[0]
                groups.append(f"{gesture}_{base_name}")

# Ensure groups and features align
    assert len(groups) == len(X), "Group count mismatch — check data loading logic."

# --- Split ensuring same group never in both sets ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    num_classes = len(label_map)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # --- Build Enhanced LSTM Model ---
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(Config.SEQUENCE_LENGTH, nfeatures), dropout=0.4, recurrent_dropout=0.2),
        LSTM(64, return_sequences=False, dropout=0.4, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.6),  # slightly increased dropout for better generalization
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary(print_fn=logging.info)

    # --- Train ---
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=40,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    # --- Evaluate ---
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"✅ Validation Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=gestures))

    # --- Save model and artifacts ---
    model.save("gesture_lstm_model.h5")
    with open(Config.LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)
    with open("scaler_lstm.pkl", "wb") as f:
        pickle.dump(scaler, f)

    logging.info("💾 Model saved as 'gesture_lstm_model.h5'")
    logging.info("--- TRAINING COMPLETE ---")


if __name__ == "__main__":
    main()
