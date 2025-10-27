from config import Config
import os, numpy as np, pickle, logging

def run_model_training():
    """Loads the collected enhanced data, trains the advanced model, and saves the best version."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    logging.info("--- Starting Enhanced Model Training ---")
    
    if not os.path.exists(Config.DATA_PATH):
        logging.error(f"Data path '{Config.DATA_PATH}' not found. Please run 'collect' first.")
        return
        
    gestures = sorted([d for d in os.listdir(Config.DATA_PATH) if os.path.isdir(os.path.join(Config.DATA_PATH, d))])
    
    if len(gestures) < 2:
        logging.warning("Please collect at least 2 different gestures before training.")
        return

    label_map = {label: idx for idx, label in enumerate(gestures)}
    sequences, labels = [], []

    logging.info("Loading collected enhanced gesture data...")
    for gesture, label in label_map.items():
        gpath = os.path.join(Config.DATA_PATH, gesture)
        sample_files = [f for f in os.listdir(gpath) if f.lower().endswith(".npy")]
        if len(sample_files) < Config.MIN_SAMPLES_PER_GESTURE:
            logging.warning(f"Gesture '{gesture}' needs {Config.MIN_SAMPLES_PER_GESTURE} samples. Aborting.")
            return
        for fname in sample_files:
            sequences.append(np.load(os.path.join(gpath, fname)))
            labels.append(label)

    X = np.array(sequences, dtype=np.float32)
    labels_arr = np.array(labels)
    y = to_categorical(labels_arr).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=labels_arr)
    
    logging.info(f"📊 Training on {len(X_train)} samples for {len(label_map)} gestures...")
    
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(Config.SEQUENCE_LENGTH, X.shape[2])),
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(label_map), activation='softmax')
    ])
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()
    
    early_stopping_callback = EarlyStopping(
        monitor='val_categorical_accuracy', patience=10, verbose=1, mode='max', restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train, epochs=Config.TRAIN_EPOCHS, validation_data=(X_test, y_test), 
        batch_size=16, verbose=1, callbacks=[early_stopping_callback]
    )

    model.save(Config.MODEL_PATH)
    with open(Config.LABELS_PATH, "wb") as f: pickle.dump(label_map, f)

    logging.info("\n--- TRAINING COMPLETE ---")
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    final_accuracy = accuracy_score(y_true, y_pred)
    logging.info(f"Final Best Validation Accuracy: {final_accuracy * 100:.2f}%")
    
    print("\nClassification Report (from best model):")
    target_names = ["" for _ in range(len(label_map))]
    for name, index in label_map.items(): target_names[index] = name
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    logging.info(f"Model saved to '{Config.MODEL_PATH}'")
