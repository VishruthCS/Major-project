#
# Main Control Panel (FINAL PROFESSIONAL VERSION)
#
# This script is the single entry point to run all parts of our project.
#

import sys
import os
import multiprocessing as mp_process

# This is to handle a potential issue with TensorFlow and OpenCV on some systems
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    """Parses command-line arguments to run the correct module."""
    if len(sys.argv) < 2:
        print("\n" + "="*50)
        print("Hybrid AI Gesture Recognizer - Control Panel")
        print("="*50)
        print("Usage: python main.py [collect | train | recognize]")
        print("\n  collect   - Step 1: Start the enhanced data collection script.")
        print("  train     - Step 2: Train the model on the collected enhanced data.")
        print("  recognize - Step 3: Run the final real-time gesture recognizer.")
        print("="*50)
        return

    command = sys.argv[1].lower()

    if command == "collect":
        print("\n--- Starting Step 1: Enhanced Data Collection ---")
        try:
            import data_collection
            data_collection.run_data_collection()
        except ImportError:
            print("ERROR: 'data_collection.py' not found in the project folder.")
            
    elif command == "train":
        print("\n--- Starting Step 2: Enhanced Model Training ---")
        try:
            import train_model as model_training
            model_training.main()
        except ImportError:
             print("ERROR: 'model_training.py' not found in the project folder.")

    elif command == "recognize":
        print("\n--- Starting Step 3: Final Hybrid AI Recognizer ---")
        try:
            import gesture_recognition
            gesture_recognition.main()
        except ImportError:
            print("ERROR: 'gesture_recognition.py' not found in the project folder.")

    else:
        print(f"Unknown command: '{command}'")
        print("Please use 'collect', 'train', or 'recognize'.")

if __name__ == "__main__":
    # This is required for multiprocessing to work correctly on Windows
    mp_process.freeze_support()
    main()

