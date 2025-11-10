#
# workers.py — Background workers (updated for Enhanced LSTM)
#

import numpy as np
import pickle
import queue
import requests
import time
from config import Config

# --- AI Model Worker ---
def model_worker(in_q, out_q, model_path, labels_path, config_dict):
    """
    TensorFlow worker that runs in a separate process.
    Loads the Enhanced LSTM model (.h5) and performs predictions.
    """
    print("[AI Worker] Starting LSTM model worker...")

    try:
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        with open(labels_path, "rb") as f:
            label_map = pickle.load(f)
        inv_label_map = {v: k for k, v in label_map.items()}
        print(f"[AI Worker] ✅ Loaded model with {len(label_map)} classes.")
    except Exception as e:
        print(f"[AI Worker] ❌ Error loading model: {e}")
        out_q.put(f"ERROR: {e}")
        return

    expected_features = 774
    sequence_length = config_dict.get("SEQUENCE_LENGTH", 30)

    while True:
        sequence = in_q.get()
        if sequence is None:
            break
        try:
            inp = np.expand_dims(sequence, axis=0)
            preds = model.predict(inp, verbose=0)[0]
            best_idx, best_prob = int(np.argmax(preds)), float(np.max(preds))
            gesture = inv_label_map.get(best_idx, "Unknown")
            out_q.put((gesture, best_prob))
        except Exception as e:
            print(f"[AI Worker] Prediction failed: {e}")
            out_q.put(("ERROR", 0.0))

# --- Gemini Worker (unchanged) ---
def gemini_worker(q_in, q_out):
    API_KEY = Config.GEMINI_API_KEY
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}"
    while True:
        keywords = q_in.get()
        if keywords is None:
            q_in.task_done()
            break
        if not keywords:
            q_out.put("")
            q_in.task_done()
            continue
        prompt = f"Form a grammatically correct English sentence from these keywords: {' '.join(keywords)}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(api_url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            sentence = result["candidates"][0]["content"]["parts"][0]["text"]
            q_out.put(sentence.strip())
        except Exception as e:
            print(f"[Gemini Worker] Error: {e}")
            q_out.put("API Error.")
        q_in.task_done()
