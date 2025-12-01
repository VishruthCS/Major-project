#
# Data Augmentation Script for Gesture Dataset
# --------------------------------------------
# ✅ Adds small variations to your .npy sequences
# ✅ Doubles or triples dataset size
# ✅ Keeps sequence shape (30, 774)
#

import os
import numpy as np
from config import Config

# --- Augmentation functions ---
def add_noise(data, noise_level=0.01):
    """Add small Gaussian noise to all features."""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def time_shift(data, shift_max=2):
    """Shift the sequence slightly in time (±2 frames)."""
    shift = np.random.randint(-shift_max, shift_max)
    if shift == 0:
        return data
    elif shift > 0:
        return np.concatenate((data[shift:], np.tile(data[-1:], (shift, 1))), axis=0)
    else:
        return np.concatenate((np.tile(data[:1], (-shift, 1)), data[:shift]), axis=0)

def scale_features(data, scale_range=(0.95, 1.05)):
    """Scale all feature values slightly."""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale

def augment_sequence(data):
    """Apply random combination of augmentations."""
    aug = data.copy()
    if np.random.rand() < 0.5:
        aug = add_noise(aug, noise_level=0.01)
    if np.random.rand() < 0.5:
        aug = time_shift(aug)
    if np.random.rand() < 0.5:
        aug = scale_features(aug)
    return aug


# --- Main augmentation loop ---
def augment_dataset():
    print("🚀 Starting gesture data augmentation...")
    base_path = Config.DATA_PATH
    augmented_count = 0

    for gesture in os.listdir(base_path):
        gesture_path = os.path.join(base_path, gesture)
        if not os.path.isdir(gesture_path):
            continue

        files = [f for f in os.listdir(gesture_path) if f.endswith('.npy')]
        if len(files) == 0:
            continue

        print(f"\n🖐️ Augmenting gesture: {gesture} ({len(files)} base samples)")
        for file in files:
            path = os.path.join(gesture_path, file)
            data = np.load(path)
            if data.shape != (Config.SEQUENCE_LENGTH, 774):
                print(f"⚠️ Skipped {file} (invalid shape {data.shape})")
                continue

            # Generate 1–2 augmented versions per file
            for i in range(1):
                aug_data = augment_sequence(data)
                new_name = file.replace('.npy', f'_aug{i}.npy')
                np.save(os.path.join(gesture_path, new_name), aug_data)
                augmented_count += 1

    print(f"\n✅ Augmentation complete. Generated {augmented_count} new samples total.")
    print("📁 All augmented files saved alongside originals in gesture folders.")


if __name__ == "__main__":
    augment_dataset()
