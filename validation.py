# validate_npy.py
import os, sys, numpy as np

DATA_DIR = "gesture_data"   # put your root dataset folder here (or "enhanced_gesture_data")
EXPECTED_FEATURES = 774     # set to 774 (your enhanced extractor)
EXPECTED_SEQ = None         # set to None to accept any seq length, or set to 30 or 40 as needed

def check_file(path):
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as e:
        return False, f"LOAD_ERROR: {e}"

    if not isinstance(arr, np.ndarray):
        return False, f"NOT_NDARRAY: {type(arr)}"

    if arr.ndim != 2:
        return False, f"BAD_DIM: ndim={arr.ndim}"

    seq, feat = arr.shape
    if EXPECTED_FEATURES and feat != EXPECTED_FEATURES:
        return False, f"BAD_FEATURES: got {feat}, expected {EXPECTED_FEATURES}"

    if EXPECTED_SEQ and seq != EXPECTED_SEQ:
        # we still accept, but return WARN
        return None, f"SEQ_MISMATCH: got {seq}, expected {EXPECTED_SEQ}"

    if not np.isfinite(arr).all():
        return False, "HAS_NAN_OR_INF"

    # constant frame check (low variance across frames)
    per_feature_std = np.std(arr, axis=0)
    if np.all(per_feature_std < 1e-6):
        return False, "CONSTANT_FRAMES"

    # near-constant: warn if most features have tiny variance
    tiny = (per_feature_std < 1e-4).sum()
    if tiny / arr.shape[1] > 0.6:
        return None, f"LOW_VARIANCE: {tiny}/{arr.shape[1]} features tiny variance"

    return True, f"OK (shape={arr.shape}, dtype={arr.dtype})"

if __name__ == "__main__":
    root = DATA_DIR
    if not os.path.exists(root):
        print("ERROR: dataset folder not found:", root); sys.exit(1)

    total = 0
    ok = 0
    warn = 0
    bad = 0
    for g in sorted(os.listdir(root)):
        gpath = os.path.join(root, g)
        if not os.path.isdir(gpath): continue
        files = [f for f in os.listdir(gpath) if f.endswith(".npy")]
        print(f"\n=== Gesture: {g} ({len(files)} files) ===")
        for f in sorted(files):
            total += 1
            path = os.path.join(gpath, f)
            status, msg = check_file(path)
            if status is True:
                ok += 1
                print(f"[OK]  {f} -> {msg}")
            elif status is None:
                warn += 1
                print(f"[WARN] {f} -> {msg}")
            else:
                bad += 1
                print(f"[BAD]  {f} -> {msg}")
    print("\nSUMMARY:", f"total={total}", f"ok={ok}", f"warn={warn}", f"bad={bad}")
