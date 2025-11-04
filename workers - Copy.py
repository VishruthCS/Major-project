#
# workers.py (Final Professional Version)
#
# This file contains all the background workers for our application,
# including the AI model worker, the TTS worker, and the Gemini worker.
#

import numpy as np
import pickle
import queue
import requests
import time

# Import from our shared files
from config import Config

# --- AI Model Worker (Runs in a separate process) ---
def model_worker(in_q, out_q, model_path, labels_path, config_dict):
    """
    This function runs in a completely separate process.
    It loads the TensorFlow model and waits for data to predict.
    """
    print("[AI Worker] Starting...")
    try:
        # Import TensorFlow locally in this process
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        with open(labels_path, "rb") as f:
            label_map = pickle.load(f)
        
        # Calculate expected feature size (base + velocity + accel)
        expected_features = (33 * 4 + 21 * 3 + 21 * 3) * 3 
        model_input_features = model.input_shape[2]
        
        if model_input_features != expected_features:
            print(f"[AI Worker] ERROR: Model input size ({model_input_features}) does not match feature extractor size ({expected_features}).")
            print("[AI Worker] Please retrain the model with the latest 'main.py train' script.")
            out_q.put(f"ERROR: Model/data mismatch.")
            return

        # Pre-warm the model
        dummy_input = np.zeros((1, config_dict['SEQUENCE_LENGTH'], expected_features))
        model.predict(dummy_input, verbose=0)
        print("[AI Worker] Model loaded and ready.")
        out_q.put("READY")

    except Exception as e:
        print(f"[AI Worker] Failed to load model: {e}")
        out_q.put(f"ERROR: {e}")
        return

    # Create the inverted map once
    inv_label_map = {v: k for k, v in label_map.items()}

    while True:
        sequence = in_q.get()
        if sequence is None: break 

        try:
            inp = np.expand_dims(sequence, axis=0)
            probs = model.predict(inp, verbose=0)[0]
            best_idx, best_prob = np.argmax(probs), np.max(probs)
            prediction = inv_label_map.get(best_idx, "Unknown")
            
            out_q.put((prediction, float(best_prob)))
        except Exception as e:
            print(f"[AI Worker] Prediction failed: {e}")
            out_q.put(("ERROR", 0.0))

# --- Text-to-Speech Worker (Runs in a separate thread) ---
def tts_worker(q, engine):
    """Worker thread for text-to-speech."""
    while True:
        text = q.get()
        if text is None: 
            q.task_done()
            break
        if text:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
        q.task_done()

# --- Gemini API Worker (Runs in a separate thread) ---
def gemini_worker(q_in, q_out):
    """Worker thread to process keywords with the Gemini API."""
    while True:
        keywords = q_in.get()
        if keywords is None: 
            q_in.task_done()
            break
        
        if not keywords or not Config.GEMINI_API_KEY or Config.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
            q_out.put("API Key Not Set" if not Config.GEMINI_API_KEY or Config.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" else "")
            q_in.task_done()
            continue

        prompt = f"The following are keywords from a signed sentence. Form a complete, grammatically correct English sentence. Do not add any extra information. Keywords: {' '.join(keywords)}"
        api_url = f"https.generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={Config.GEMINI_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        try:
            response = requests.post(api_url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            completed_sentence = result['candidates'][0]['content']['parts'][0]['text']
            q_out.put(completed_sentence.strip())
        except Exception:
            q_out.put("API Error.")
        q_in.task_done()
