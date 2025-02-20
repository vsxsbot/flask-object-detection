import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)
camera = cv2.VideoCapture(0)

object_detected = False  # Global flag

def detect_object(frame):
    global object_detected
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Check for objects (Modify with actual object detection logic)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_detected = len(contours) > 0  # Set flag based on contours

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            detect_object(frame)  # Call detection function
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_object')
def check_object():
    return jsonify({"object_detected": object_detected})

@app.route('/capture')
def capture():
    success, frame = camera.read()
    if success:
        cv2.imwrite("captured_image.jpg", frame)
    return jsonify({"message": "Image Captured"})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
