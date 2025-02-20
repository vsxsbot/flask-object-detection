from flask import Flask, render_template, Response, request
import cv2
import numpy as np

app = Flask(__name__)

def process_frame(frame):
    # Your object detection code here (modify as needed)
    return frame  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    file = request.files['video']
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    processed_frame = process_frame(frame)
    
    _, buffer = cv2.imencode('.jpg', processed_frame)
    return buffer.tobytes(), {'Content-Type': 'image/jpeg'}

if __name__ == "__main__":
    app.run(debug=True)
