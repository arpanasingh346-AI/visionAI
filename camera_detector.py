"""
VisionAI - Camera Detector & Hardware Diagnostics
Detects connected camera devices, tests resolutions, and returns hardware status.
"""
import sys
import time
import cv2

# Ensure UTF-8 output if supported
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def test_camera_index(index: int, backend=cv2.CAP_DSHOW):
    """
    Tests if a camera index is accessible and fetches its properties.
    """
    cap = cv2.VideoCapture(index, backend) if sys.platform.startswith('win') else cv2.VideoCapture(index)
    
    if not cap.isOpened():
        # Fallback to default backend if DSHOW fails
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            return None

    # Try reading a test frame
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return None

    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0  # reasonable fallback

    # Check common 16:9 or 4:3 aspect ratio
    aspect = f"{w}:{h}"
    if abs(w / h - 16 / 9) < 0.05:
        aspect = "16:9 (Widescreen)"
    elif abs(w / h - 4 / 3) < 0.05:
        aspect = "4:3 (Standard)"

    info = {
        "index": index,
        "name": f"Camera Device #{index}" + (" (Default/Built-in)" if index == 0 else " (External/Secondary)"),
        "resolution": f"{w}x{h}",
        "width": w,
        "height": h,
        "aspect_ratio": aspect,
        "fps": round(fps, 1),
        "status": "Ready",
        "backend": "DirectShow" if sys.platform.startswith('win') else "V4L2/Default"
    }
    cap.release()
    return info

def get_available_cameras(max_scan=5):
    """
    Scans indices 0 through max_scan-1 to find all accessible camera devices.
    """
    cameras = []
    for idx in range(max_scan):
        info = test_camera_index(idx)
        if info is not None:
            cameras.append(info)
    return cameras

def display_camera_report():
    """
    Prints a formatted diagnostics report to the console.
    """
    print("=" * 60)
    print("  [VisionAI] - Connected Camera Diagnostics")
    print("=" * 60)
    print("Scanning available camera devices on system...\n")
    
    start_time = time.time()
    cameras = get_available_cameras()
    duration = time.time() - start_time

    if not cameras:
        print("[!] No active cameras detected!")
        print("Tips:")
        print("  1. Make sure your USB webcam is plugged in firmly.")
        print("  2. Check if another app (Zoom, Teams, Skype, Browser) is using the camera.")
        print("  3. Verify camera privacy settings in Windows Settings > Privacy > Camera.")
    else:
        print(f"[OK] Found {len(cameras)} active camera device(s) in {duration:.2f}s:\n")
        for cam in cameras:
            print(f"  * [{cam['index']}] {cam['name']}")
            print(f"      - Status       : {cam['status']}")
            print(f"      - Resolution   : {cam['resolution']} [{cam['aspect_ratio']}]")
            print(f"      - Frame Rate   : ~{cam['fps']} FPS")
            print(f"      - API Backend  : {cam['backend']}")
            print("-" * 50)
    print("=" * 60)
    return cameras

if __name__ == "__main__":
    display_camera_report()

