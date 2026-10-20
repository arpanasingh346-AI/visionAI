"""
VisionAI - Web Application Server
Powered by Flask, OpenCV, YOLO, and Modern Glassmorphism Web Interface.
"""
import os
import sys
import time
import base64
import socket
import threading
import webbrowser
from datetime import datetime
from io import BytesIO
from pathlib import Path

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
from PIL import Image
import qrcode
from flask import Flask, render_template, Response, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

import config
from ai_engine import VisionEngine
from camera_detector import get_available_cameras, test_camera_index

# Initialize Flask App
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

USE_SSL = "--ssl" in sys.argv

# Initialize AI Vision Engine (Singleton)
print("[VisionAI Server] Initializing AI Vision Engine...")
engine = VisionEngine()

# Global camera state for live streaming
class CameraStream:
    def __init__(self):
        self.cap = None
        self.current_index = config.DEFAULT_CAMERA_INDEX
        self.is_running = False
        self.conf_threshold = config.DEFAULT_CONFIDENCE
        self.detect_faces = True
        self.selected_category = "All"

    def open(self, index=0):
        if self.cap is not None:
            self.cap.release()
        
        backend = cv2.CAP_DSHOW if sys.platform.startswith('win') else 0
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)

        if self.cap.isOpened():
            self.current_index = index
            self.is_running = True
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"[CameraStream] Camera {index} opened successfully.")
            return True
        else:
            self.is_running = False
            print(f"[CameraStream] Failed to open camera {index}.")
            return False

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            if not self.open(self.current_index):
                return None, None
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False

camera_stream = CameraStream()


@app.route("/")
def index():
    """Serves the main application web interface."""
    return render_template("index.html")


@app.route("/api/cameras", methods=["GET"])
def api_cameras():
    """Returns a list of all detected cameras and their diagnostics."""
    cams = get_available_cameras()
    return jsonify({
        "success": True,
        "count": len(cams),
        "cameras": cams,
        "current_active_index": camera_stream.current_index
    })


@app.route("/api/set_camera", methods=["POST"])
def api_set_camera():
    """Switches the active camera index."""
    data = request.get_json() or {}
    cam_index = int(data.get("index", 0))
    success = camera_stream.open(cam_index)
    return jsonify({
        "success": success,
        "camera_index": cam_index,
        "message": f"Switched to Camera #{cam_index}" if success else "Failed to open camera"
    })


@app.route("/api/settings", methods=["POST"])
def api_settings():
    """Updates live detection settings (confidence, faces, category filter)."""
    data = request.get_json() or {}
    if "confidence" in data:
        camera_stream.conf_threshold = float(data["confidence"])
    if "detect_faces" in data:
        camera_stream.detect_faces = bool(data["detect_faces"])
    if "category" in data:
        camera_stream.selected_category = str(data["category"])
    
    return jsonify({
        "success": True,
        "confidence": camera_stream.conf_threshold,
        "detect_faces": camera_stream.detect_faces,
        "category": camera_stream.selected_category
    })


def generate_live_mjpeg():
    """Generator function that yields MJPEG video stream."""
    last_time = time.time()
    while True:
        frame = camera_stream.get_frame()
        if frame is None:
            # Yield standby placeholder frame
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Connecting to Camera...", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
            _, buffer = cv2.imencode('.jpg', placeholder)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5)
            continue

        # Process frame with AI Engine
        annotated, analysis = engine.process_frame(
            frame,
            conf_threshold=camera_stream.conf_threshold,
            detect_faces=camera_stream.detect_faces,
            selected_category=camera_stream.selected_category,
            draw_annotations=True
        )

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # Target ~25-30 FPS to reduce CPU load
        time.sleep(0.02)


@app.route("/api/live_stream")
def live_stream():
    """MJPEG stream endpoint for real-time video."""
    return Response(
        generate_live_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/process_frame", methods=["POST"])
def api_process_frame():
    """
    Processes an individual base64 frame from browser webcam.
    Allows running camera directly through user's web browser without server webcam access!
    """
    try:
        data = request.get_json() or {}
        image_data = data.get("image")
        conf = float(data.get("confidence", config.DEFAULT_CONFIDENCE))
        detect_faces = bool(data.get("detect_faces", True))
        selected_category = str(data.get("category", "All"))

        if not image_data:
            return jsonify({"success": False, "error": "No image data provided"}), 400

        # Decode base64 image
        if "," in image_data:
            image_data = image_data.split(",")[1]
        decoded = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "error": "Could not decode frame"}), 400

        # Process frame
        annotated, analysis = engine.process_frame(
            frame,
            conf_threshold=conf,
            detect_faces=detect_faces,
            selected_category=selected_category,
            draw_annotations=True
        )

        # Encode annotated image back to base64
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "success": True,
            "analysis": analysis,
            "annotated_image": annotated_b64
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/identify_image", methods=["POST"])
def api_identify_image():
    """
    Endpoint for uploading an image to perform deep AI identification.
    Returns detected objects, category classification, color palette, and AI scene summary.
    """
    try:
        file = request.files.get("image")
        conf = float(request.form.get("confidence", config.DEFAULT_CONFIDENCE))
        detect_faces = request.form.get("detect_faces", "true").lower() == "true"
        category = request.form.get("category", "All")

        if not file:
            return jsonify({"success": False, "error": "No image file uploaded"}), 400

        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"success": False, "error": "Invalid image file format"}), 400

        h, w = img.shape[:2]

        # 1. AI Vision Object & Face Detection
        annotated, analysis = engine.process_frame(
            img,
            conf_threshold=conf,
            detect_faces=detect_faces,
            selected_category=category,
            draw_annotations=True
        )

        # 2. Extract Dominant Colors
        colors = engine.extract_dominant_colors(img, num_colors=5)

        # 3. Generate Natural Language AI Scene Summary
        scene_story = engine.generate_scene_story(analysis)

        # 4. Save uploaded image & result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f"identify_{timestamp}.jpg"
        save_path = config.UPLOADS_DIR / save_filename
        cv2.imwrite(str(save_path), annotated)

        # Encode both original and annotated as base64 for direct browser rendering
        _, orig_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        _, anno_buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        orig_b64 = "data:image/jpeg;base64," + base64.b64encode(orig_buf).decode('utf-8')
        anno_b64 = "data:image/jpeg;base64," + base64.b64encode(anno_buf).decode('utf-8')

        return jsonify({
            "success": True,
            "filename": file.filename,
            "dimensions": f"{w}x{h}",
            "analysis": analysis,
            "dominant_colors": colors,
            "scene_story": scene_story,
            "original_image": orig_b64,
            "annotated_image": anno_b64,
            "download_url": f"/captures/{save_filename}"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    """Captures the current frame from live stream and saves to disk."""
    frame = camera_stream.get_frame()
    if frame is None:
        return jsonify({"success": False, "error": "Could not grab frame from camera"}), 500

    annotated, analysis = engine.process_frame(
        frame,
        conf_threshold=camera_stream.conf_threshold,
        detect_faces=camera_stream.detect_faces,
        selected_category=camera_stream.selected_category,
        draw_annotations=True
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}.jpg"
    path = config.CAPTURES_DIR / filename
    cv2.imwrite(str(path), annotated)

    _, buf = cv2.imencode('.jpg', annotated)
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

    return jsonify({
        "success": True,
        "filename": filename,
        "saved_path": str(path),
        "analysis": analysis,
        "image": b64
    })


@app.route("/captures/<filename>")
def serve_capture(filename):
    """Serves captured or analyzed images."""
    if (config.CAPTURES_DIR / filename).exists():
        return send_from_directory(config.CAPTURES_DIR, filename)
    return send_from_directory(config.UPLOADS_DIR, filename)


def generate_qr_bytes(url: str, fill="#06b6d4", back="#090d16"):
    """Generates PNG bytes for a QR code."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color=fill, back_color=back)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@app.route("/api/network_info")
def api_network_info():
    """Returns local network access URLs and QR code for mobile connection."""
    net_info = config.get_network_urls(port=config.PORT, use_ssl=USE_SSL)
    mobile_url = net_info["network"]

    # Generate QR Base64
    buf = generate_qr_bytes(mobile_url)
    b64_qr = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

    return jsonify({
        "success": True,
        "local_url": net_info["localhost"],
        "mobile_url": mobile_url,
        "local_ip": net_info["local_ip"],
        "port": net_info["port"],
        "protocol": net_info["protocol"],
        "qr_code_b64": b64_qr
    })


@app.route("/api/qr_code")
def api_qr_code():
    """Serves raw PNG image of the mobile access QR code."""
    net_info = config.get_network_urls(port=config.PORT, use_ssl=USE_SSL)
    buf = generate_qr_bytes(net_info["network"])
    return send_file(buf, mimetype="image/png")


@app.route("/api/health")
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "online",
        "models": {
            "yolo": True,
            "face_detection": engine.face_detection_available
        },
        "supported_classes": len(engine.model.names)
    })


def find_available_port(start_port=5000, max_attempts=10):
    """Finds the first available TCP port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    return start_port


def launch_browser_when_ready(url, delay=1.5):
    """Opens the user's default web browser once Flask is up and running."""
    def _open():
        time.sleep(delay)
        try:
            print(f"[VisionAI] Opening web browser at {url} ...")
            webbrowser.open_new_tab(url)
        except Exception as e:
            print(f"[VisionAI] Notice: Please open {url} manually in your browser.")

    t = threading.Thread(target=_open, daemon=True)
    t.start()


if __name__ == "__main__":
    active_port = find_available_port(config.PORT)
    config.PORT = active_port
    net_info = config.get_network_urls(port=active_port, use_ssl=USE_SSL)
    local_url = net_info["localhost"]
    mobile_url = net_info["network"]

    print("=" * 65)
    print("      🚀 VisionAI - Web & Mobile Live Studio")
    print("=" * 65)
    print(f"  🖥️  Desktop Access : {local_url}")
    print(f"  📱 Mobile Access  : {mobile_url}")
    print("=" * 65)
    print("  📲 Scan with your Phone Camera to open on Mobile:")
    try:
        # Print terminal ASCII QR code
        qr = qrcode.QRCode(box_size=1, border=1)
        qr.add_data(mobile_url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception:
        pass
    print("=" * 65)
    print("Press Ctrl+C to stop server.\n")

    # Automatically launch browser when server is ready
    if "--no-browser" not in sys.argv:
        launch_browser_when_ready(local_url, delay=1.5)

    ssl_ctx = "adhoc" if USE_SSL else None
    app.run(host=config.HOST, port=active_port, debug=config.DEBUG, ssl_context=ssl_ctx)
