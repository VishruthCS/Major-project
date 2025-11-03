import cv2

def test_camera():
    """Test script to check camera access on different indices."""
    for i in range(3):  # Try indices 0, 1, 2
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Camera accessible on index {i}")
            ret, frame = cap.read()
            if ret:
                print(f"✅ Successfully captured frame on index {i}")
            else:
                print(f"❌ Could not capture frame on index {i}")
            cap.release()
            return i  # Return the working index
        else:
            print(f"❌ Camera not accessible on index {i}")
    print("❌ No camera found on indices 0-2")
    return None

if __name__ == "__main__":
    working_index = test_camera()
    if working_index is not None:
        print(f"Use camera index {working_index} in your scripts.")
    else:
        print("Check camera permissions, connections, or if another app is using it.")
