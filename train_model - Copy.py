#
# Step 2: Model Training (FINAL STABLE VERSION)
#
# This script trains a fast and accurate scikit-learn model.
# It does NOT use TensorFlow, guaranteeing no camera conflicts.
#

import os
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Import from our shared files
from config import Config

def main():
    """Loads collected data, flattens it, and trains a KNN model."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    logging.info("--- Starting Stable Model Training ---")
    
    if not os.path.exists(Config.DATA_PATH):
        logging.error(f"Data path '{Config.DATA_PATH}' not found. Please run 'python main.py collect' first.")
        return
        
    gestures = sorted([d for d in os.listdir(Config.DATA_PATH) if os.path.isdir(os.path.join(Config.DATA_PATH, d))])
    
    if len(gestures) < 2:
        logging.warning("Please collect data for at least 2 different gestures before training.")
        return

    label_map = {label: idx for idx, label in enumerate(gestures)}
    features, labels = [], []

    logging.info("Loading collected enhanced gesture data...")
    for gesture, label in label_map.items():
        gpath = os.path.join(Config.DATA_PATH, gesture)
        sample_files = [f for f in os.listdir(gpath) if f.lower().endswith(".npy")]
        if len(sample_files) < Config.MIN_SAMPLES_PER_GESTURE:
            logging.warning(f"Gesture '{gesture}' needs {Config.MIN_SAMPLES_PER_GESTURE} samples. Aborting.")
            return
        for fname in sample_files:
            # We load the sequence, but flatten it for the static classifier
            sequence = np.load(os.path.join(gpath, fname))
            # We flatten the 30-frame sequence into a single feature vector
            features.append(sequence.flatten()) 
            labels.append(label)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logging.info(f"📊 Training on {len(X_train)} samples for {len(label_map)} gestures...")
    
    # Scale the data (very important for KNN)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = KNeighborsClassifier(n_neighbors=min(5, len(X_train)))
    
    model.fit(X_train, y_train)

    # Save the model, the label map, AND the scaler
    model_data = {
        'model': model,
        'label_map': {idx: label for label, idx in label_map.items()}, # Save in {index: name} format
        'scaler': scaler
    }
    
    with open(Config.MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    
    # Save the label map separately (good practice)
    with open(Config.LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)

    logging.info("\n--- TRAINING COMPLETE ---")
    
    y_pred = model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Final Best Validation Accuracy: {final_accuracy * 100:.2f}%")
    
    print("\nClassification Report (from best model):")
    print(classification_report(y_test, y_pred, target_names=gestures, zero_division=0))
    
    logging.info(f"Model saved to '{Config.MODEL_PATH}'")

if __name__ == "__main__":
    main()

