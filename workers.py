import queue
import logging
import requests
import pyttsx3
from config import Config

def tts_worker(q, engine):
    while True:
        text = q.get()
        if text is None: break
        if text:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logging.error(f"TTS Error: {e}")
        q.task_done()

def gemini_worker(q_in, q_out):
    if not Config.GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not found.")
        return

    while True:
        keywords = q_in.get()
        if keywords is None: break

        if not keywords:
            q_out.put("No keywords provided.")
            q_in.task_done()
            continue

        prompt = (
            f"Form a grammatically correct English sentence from these sign keywords:\n"
            f"{' '.join(keywords)}"
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        try:
            response = requests.post(Config.GEMINI_URL, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            sentence = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            q_out.put(sentence)
        except requests.exceptions.Timeout:
            q_out.put("Gemini API timeout — try again.")
        except Exception as e:
            q_out.put(f"API Error: {e}")
        finally:
            q_in.task_done()
