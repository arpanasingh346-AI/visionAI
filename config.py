"""
VisionAI - Configuration Module
Defines settings, paths, categories, and thresholds.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Model and asset paths
MODEL_PATH = str(BASE_DIR / "yolo26n.pt")
FACE_CASCADE_PATH = str(BASE_DIR / "haarcascade_frontalface_default.xml")

# Storage directories
CAPTURES_DIR = BASE_DIR / "captures"
UPLOADS_DIR = BASE_DIR / "uploads"
CAPTURES_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Web server settings
HOST = "0.0.0.0"
PORT = 5000
DEBUG = False

def get_local_ip() -> str:
    """Returns the local LAN IP address of this machine."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Does not actually send data over network
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def get_network_urls(port: int = PORT, use_ssl: bool = False):
    """Returns local and network access URLs."""
    protocol = "https" if use_ssl else "http"
    local_ip = get_local_ip()
    return {
        "localhost": f"{protocol}://127.0.0.1:{port}",
        "network": f"{protocol}://{local_ip}:{port}",
        "local_ip": local_ip,
        "port": port,
        "protocol": protocol
    }

# Vision settings
DEFAULT_CONFIDENCE = 0.35
DEFAULT_IMGSZ = 416
DEFAULT_CAMERA_INDEX = 0

# Category grouping for the 80 COCO classes
CATEGORY_MAP = {
    # People & Faces
    "person": "Human",
    
    # Vehicles & Transportation
    "bicycle": "Vehicles", "car": "Vehicles", "motorcycle": "Vehicles", "airplane": "Vehicles",
    "bus": "Vehicles", "train": "Vehicles", "truck": "Vehicles", "boat": "Vehicles",
    
    # Outdoor & Traffic
    "traffic light": "Outdoor", "fire hydrant": "Outdoor", "stop sign": "Outdoor", 
    "parking meter": "Outdoor", "bench": "Outdoor",
    
    # Animals
    "bird": "Animals", "cat": "Animals", "dog": "Animals", "horse": "Animals", 
    "sheep": "Animals", "cow": "Animals", "elephant": "Animals", "bear": "Animals", 
    "zebra": "Animals", "giraffe": "Animals",
    
    # Accessories & Personal items
    "backpack": "Accessories", "umbrella": "Accessories", "handbag": "Accessories", 
    "tie": "Accessories", "suitcase": "Accessories",
    
    # Sports equipment
    "frisbee": "Sports", "skis": "Sports", "snowboard": "Sports", "sports ball": "Sports",
    "kite": "Sports", "baseball bat": "Sports", "baseball glove": "Sports", "skateboard": "Sports",
    "surfboard": "Sports", "tennis racket": "Sports",
    
    # Kitchen & Dining
    "bottle": "Dining", "wine glass": "Dining", "cup": "Dining", "fork": "Dining",
    "knife": "Dining", "spoon": "Dining", "bowl": "Dining", "banana": "Food",
    "apple": "Food", "sandwich": "Food", "orange": "Food", "broccoli": "Food",
    "carrot": "Food", "hot dog": "Food", "pizza": "Food", "donut": "Food", "cake": "Food",
    
    # Furniture & Home
    "chair": "Furniture", "couch": "Furniture", "potted plant": "Home", "bed": "Furniture",
    "dining table": "Furniture", "toilet": "Home",
    
    # Electronics & Technology
    "tv": "Electronics", "laptop": "Electronics", "mouse": "Electronics", "remote": "Electronics",
    "keyboard": "Electronics", "cell phone": "Electronics", "microwave": "Electronics",
    "oven": "Electronics", "toaster": "Electronics", "sink": "Home", "refrigerator": "Electronics",
    "book": "Education", "clock": "Home", "vase": "Home", "scissors": "Tools",
    "teddy bear": "Toys", "hair drier": "Personal Care", "toothbrush": "Personal Care"
}

# Color palette (BGR for OpenCV, Hex for Web)
CATEGORY_COLORS_BGR = {
    "Human": (0, 220, 255),       # Vibrant Yellow-Orange
    "Vehicles": (255, 140, 0),     # Deep Orange/Blue
    "Animals": (0, 255, 127),      # Emerald Green
    "Electronics": (255, 0, 128),  # Electric Purple
    "Food": (0, 165, 255),         # Orange
    "Dining": (255, 191, 0),       # Light Blue
    "Furniture": (180, 105, 255),  # Pink/Lavender
    "Outdoor": (128, 128, 128),    # Gray
    "Accessories": (255, 215, 0),  # Gold
    "Sports": (50, 205, 50),       # Lime Green
    "Home": (221, 160, 221),       # Plum
    "Tools": (192, 192, 192),      # Silver
    "Toys": (255, 105, 180),       # Hot Pink
    "Education": (135, 206, 235),  # Sky Blue
    "Personal Care": (240, 128, 128), # Coral
    "Default": (0, 255, 0)         # Green
}
