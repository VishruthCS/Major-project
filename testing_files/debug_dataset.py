# debug_dataset.py
import os
import numpy as np
from config import Config

def analyze_motion():
    print("--- 🔍 Analyzing Dataset Motion Patterns ---")
    if not os.path.exists(Config.DATA_PATH):
        print(f"❌ Data path '{Config.DATA_PATH}' not found.")
        return

    stats = {}

    # Indices for Enhanced Features (774 total)
    # 0-257: Raw Pos (Static)
    # 258-515: Velocity (Motion)
    # 516-773: Acceleration (Change in Motion)
    
    # We specifically look at Right Hand Velocity (Indices 453 to 515)
    # Calculation: 258 (Start of Vel) + 195 (Start of RH in Raw) = 453
    rh_vel_start = 453
    rh_vel_end = 516

    for gesture in os.listdir(Config.DATA_PATH):
        g_path = os.path.join(Config.DATA_PATH, gesture)
        if not os.path.isdir(g_path): continue
        
        motion_scores = []
        files = [f for f in os.listdir(g_path) if f.endswith('.npy')]
        
        for f in files:
            try:
                data = np.load(os.path.join(g_path, f))
                # Average absolute velocity of the Right Hand across all frames
                # Shape: (frames, features)
                rh_velocity = data[:, rh_vel_start:rh_vel_end]
                avg_motion = np.mean(np.abs(rh_velocity))
                motion_scores.append(avg_motion)
            except:
                pass
        
        if motion_scores:
            avg_score = np.mean(motion_scores)
            stats[gesture] = avg_score
            print(f"📁 {gesture:12s} | Samples: {len(files):3d} | Avg Hand Motion: {avg_score:.5f}")

    print("\n--- ANALYSIS ---")
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_stats) >= 2:
        g1, s1 = sorted_stats[0]
        g2, s2 = sorted_stats[1]
        diff_percent = abs(s1 - s2) / ((s1 + s2) / 2) * 100
        
        print(f"Most Active:  {g1} ({s1:.5f})")
        print(f"Least Active: {g2} ({s2:.5f})")
        print(f"Difference:   {diff_percent:.1f}%")
        
        if diff_percent < 15.0:
            print("\n⚠️ CRITICAL WARNING: Your gestures have very similar motion intensity!")
            print("   The model cannot distinguish them based on speed/movement.")
            print("   FIX: Re-record with EXAGGERATED movements.")
        else:
            print("\n✅ Motion intensity is distinct. The issue might be spatial (hand position).")

if __name__ == "__main__":
    analyze_motion()