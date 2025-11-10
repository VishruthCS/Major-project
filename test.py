# import os
# from config import Config

# counts = {}
# for gesture in os.listdir(Config.DATA_PATH):
#     path = os.path.join(Config.DATA_PATH, gesture)
#     if os.path.isdir(path):
#         counts[gesture] = len([f for f in os.listdir(path) if f.endswith('.npy')])
# print("Sample counts per gesture:")
# for g, c in counts.items():
#     print(f"{g:12s} → {c}")
# import shutil, random, os
# src = os.path.join(Config.DATA_PATH, "vishruth")
# files = [os.path.join(src, f) for f in os.listdir(src) if f.endswith('.npy')]
# for i in range(10):  # duplicate to balance
#     shutil.copy(random.choice(files), os.path.join(src, f"dup_{i}.npy"))
# import numpy as np, os

# base = "enhanced_gesture_data"
# for gesture in os.listdir(base):
#     gesture_path = os.path.join(base, gesture)
#     if os.path.isdir(gesture_path):
#         for f in os.listdir(gesture_path):
#             if f.endswith(".npy"):
#                 arr = np.load(os.path.join(gesture_path, f))
#                 if arr.shape != (30, 774) and arr.shape != (30, 2322):
#                     print(f"⚠️ Invalid shape {arr.shape} in {gesture}/{f}")
# import os
# import numpy as np

# # ✅ Path to your gesture data
# base_path = "enhanced_gesture_data"

# # ✅ Valid shapes your system supports
# valid_shapes = {(30, 774), (30, 2322)}

# print(f"🔍 Scanning '{base_path}' for corrupted .npy files...\n")
# corrupted_files = []

# for gesture in os.listdir(base_path):
#     gesture_dir = os.path.join(base_path, gesture)
#     if not os.path.isdir(gesture_dir):
#         continue

#     for file in os.listdir(gesture_dir):
#         if file.endswith(".npy"):
#             fpath = os.path.join(gesture_dir, file)
#             try:
#                 arr = np.load(fpath)
#                 if arr.shape not in valid_shapes:
#                     print(f"⚠️ Shape mismatch {arr.shape} → {gesture}/{file}")
#                     corrupted_files.append(fpath)
#             except Exception as e:
#                 print(f"❌ Corrupted or unreadable: {gesture}/{file} ({e})")
#                 corrupted_files.append(fpath)

# print("\n------------------------------------------")
# if corrupted_files:
#     print(f"⚠️ Found {len(corrupted_files)} corrupted or invalid files.")
#     choice = input("🗑️ Do you want to delete them? (y/n): ").strip().lower()
#     if choice == 'y':
#         for f in corrupted_files:
#             os.remove(f)
#             print(f"🗑️ Deleted: {f}")
#     else:
#         print("❎ Skipped deletion.")
# else:
#     print("✅ All .npy files are valid and ready for training!")
import numpy as np, os
base = "enhanced_gesture_data"  # adjust if different
for g in os.listdir(base):
    p = os.path.join(base, g)
    for f in os.listdir(p):
        if f.endswith(".npy"):
            arr = np.load(os.path.join(p, f))
            print(f"{g}/{f} → {arr.shape}")
            break
