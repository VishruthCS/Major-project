import sys, logging
from data_collection import run_data_collection
from train_model import run_model_training
from recognize import run_gesture_recognition

def main_control():
    if len(sys.argv) < 2:
        print("\n" + "="*50)
        print("Hybrid AI Gesture Recognizer - Control Panel")
        print("="*50)
        print("Usage: python main.py [collect | train | recognize]")
        print("="*50)
        return

    command = sys.argv[1].lower()
    if command == "collect":
        run_data_collection()
    elif command == "train":
        run_model_training()
    elif command == "recognize":
        run_gesture_recognition()
    else:
        print(f"Unknown command: '{command}'")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main_control()
