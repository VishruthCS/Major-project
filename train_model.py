# train_model.py (FIXED)
import os, json, logging, pickle, random, numpy as np
from unicodedata import bidirectional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Multiply, Softmax, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Bidirectional
from config import Config

# Fix for potential OpenMP runtime duplicate error on some systems
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def stable_attention_block(inputs):
    # inputs: (batch, time, features)
    att = Dense(1, activation='tanh')(inputs)          # (batch, time, 1)
    att = Softmax(axis=1)(att)                         # normalize across time
    context = Multiply()([inputs, att])                # (batch, time, features)
    pooled = GlobalAveragePooling1D()(context)         # (batch, features)
    return pooled

def augment_sequence(seq):
    # Simple augmentations: jitter, time crop/pad
    seq = seq.copy()
    # jitter: small gaussian noise
    noise = np.random.normal(0, 1e-3, seq.shape)
    seq = seq + noise
    # time warp: random repeat or drop frames
    if random.random() < 0.3:
        factor = random.uniform(0.92, 1.08)
        indices = np.clip((np.arange(seq.shape[0]) * factor).astype(int), 0, seq.shape[0]-1)
        seq = seq[indices]
    return seq

def load_data():
    gestures = sorted([d for d in os.listdir(Config.DATA_PATH)
                       if os.path.isdir(os.path.join(Config.DATA_PATH, d)) and not d.startswith("_")])
    label_map = {g:i for i,g in enumerate(gestures)}
    features, labels, groups = [], [], []
    logging.info("📂 Loading gesture sequences...")
    for g,idx in label_map.items():
        gdir = os.path.join(Config.DATA_PATH, g)
        for fn in os.listdir(gdir):
            if not fn.endswith(".npy"): continue
            seq = np.load(os.path.join(gdir, fn)).astype(np.float32)
            
            # --- CHECK 1: Valid Shape ---
            # Data from features.py is ALREADY 774 dims (Pos + Vel + Acc)
            if seq.ndim != 2 or seq.shape[1] != 774:
                logging.warning(f"⚠️ Skipping {g}/{fn} shape={seq.shape}")
                continue
            
            # --- CHECK 2: Pad/Trim to Config.SEQUENCE_LENGTH ---
            if seq.shape[0] < Config.SEQUENCE_LENGTH:
                pad = np.tile(seq[-1:], (Config.SEQUENCE_LENGTH - seq.shape[0], 1))
                seq = np.concatenate([seq, pad], axis=0)
            elif seq.shape[0] > Config.SEQUENCE_LENGTH:
                seq = seq[:Config.SEQUENCE_LENGTH]
            
            features.append(seq)
            labels.append(idx)
            
            # Grouping to prevent leakage
            base_name = fn.split('_aug')[0].replace('.npy', '')
            groups.append(f"{g}_{base_name}")

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    logging.info(f"✅ Loaded {len(X)} sequences across {len(label_map)} gestures")
    return X, y, label_map, groups

def preprocess_and_scale(X, scaler=None, fit_scaler=False):
    # Standardize features by removing the mean and scaling to unit variance
    ns, tfm, feat = X.shape
    X_reshaped = X.reshape(ns * tfm, feat)
    
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
    else:
        if scaler is None:
            raise ValueError("scaler is None but fit_scaler is False")
        X_scaled = scaler.transform(X_reshaped)
        
    X = X_scaled.reshape(ns, tfm, feat)
    return X, scaler

def build_model(seq_len, feat_dim, num_classes):
    inp = Input(shape=(seq_len, feat_dim), name="input_layer")
    
    # LSTM Layers
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.4, recurrent_dropout=0.2))(inp)
    
    # Attention
    x = stable_attention_block(x)
    
    # Classification Head
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    logging.info("--- 🚀 Starting Hybrid LSTM + Attention Training ---")
    if not os.path.exists(Config.DATA_PATH):
        logging.error(f"❌ Data path '{Config.DATA_PATH}' missing.")
        return
        
    X, y, label_map, groups = load_data()
    if len(X) == 0:
        logging.error("❌ No valid samples found.")
        return

    # REMOVED: add_dynamics(X) 
    # Why? features.py already calculates Velocity and Acceleration (774 dims).
    # Adding it again corrupts the data and breaks compatibility with gesture_recognition.py.

    # GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale Data
    X_train, scaler = preprocess_and_scale(X_train_raw, scaler=None, fit_scaler=True)
    X_test, _ = preprocess_and_scale(X_test_raw, scaler=scaler, fit_scaler=False)

    # Prepare Labels
    num_classes = len(label_map)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Class Weights
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = {i: float(w) for i,w in enumerate(cw)}
    logging.info(f"Class weights: {class_weight}")

    # Build Model
    model = build_model(Config.SEQUENCE_LENGTH, X_train.shape[2], num_classes)
    model.summary(print_fn=logging.info)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint("models/gesture_lstm_model.keras", save_best_only=True, monitor='val_loss')
    ]

    # On-the-fly augmentation
    aug_X, aug_y = [], []
    for i in range(len(X_train)):
        seq = X_train[i]
        if random.random() < 0.6: # Augment 60% of training data
            aug_seq = augment_sequence(seq)
            # Resize logic in case augmentation changed length
            if aug_seq.shape[0] < Config.SEQUENCE_LENGTH:
                pad = np.tile(aug_seq[-1:], (Config.SEQUENCE_LENGTH - aug_seq.shape[0], 1))
                aug_seq = np.concatenate([aug_seq, pad], axis=0)
            elif aug_seq.shape[0] > Config.SEQUENCE_LENGTH:
                aug_seq = aug_seq[:Config.SEQUENCE_LENGTH]
            aug_X.append(aug_seq)
            aug_y.append(y_train[i])
            
    if aug_X:
        X_train = np.concatenate([X_train, np.array(aug_X, dtype=np.float32)], axis=0)
        y_train_cat = to_categorical(np.concatenate([y_train, np.array(aug_y, dtype=np.int32)]), num_classes)

    logging.info(f"Train samples: {len(X_train)} | Val samples: {len(X_test)}")

    # Train
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=60,
        batch_size=16,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    # Evaluate
    preds = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, preds)
    logging.info(f"✅ Validation Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, preds, target_names=list(label_map.keys()), zero_division=0))

    # Save
    model.save("models/gesture_lstm_model.keras")
    with open("models/scaler_lstm.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(Config.LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)

    metadata = {
        "sequence_length": Config.SEQUENCE_LENGTH,
        "feature_dim": X_train.shape[2],
        "model_file": "models/gesture_lstm_model.keras"
    }
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info("💾 Model saved to 'models/gesture_lstm_model.keras' and metadata written to models/model_metadata.json")

if __name__ == "__main__":
    main()