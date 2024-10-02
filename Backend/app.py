from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import asyncio
import websockets
import base64  # Base64 인코딩을 위해 필요
from ultralytics import YOLO

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')  # Use 'eventlet' or 'gevent' for compatibility with Flask-SocketIO

# Load the YOLOv8 model
model = YOLO('./runs/detect/train3/weights/best.pt')
class_names = model.names

# Constants for image processing
center_x = 320
center_y = 640
img_width = 640
img_height = 640
area_width = img_width // 3
area1_range = (0, area_width)
area2_range = (area_width, area_width * 2)
area3_range = (area_width * 2, img_width)
jetracer_value = -90

receive_images = None

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

# WebSocket to receive image data
async def async_receive_image():
    global receive_images
    uri = "ws://10.1.80.245:5000"
    print("Connecting to WebSocket...")
    async with websockets.connect(uri) as websocket:
        while True:
            # 수신된 스트리밍 데이터 처리
            data = await websocket.recv()
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            receive_images = img
            
            # 이미지를 제대로 수신했는지 확인
            print("Received an image frame")

            # 이미지를 Base64로 인코딩하여 전송
            _, jpeg = cv2.imencode('.jpg', img)
            frame = base64.b64encode(jpeg.tobytes()).decode('utf-8')  # Base64 인코딩
            socketio.emit('image_update', frame)  # 이미지 업데이트 이벤트 전송


def receive_image():
    # This function runs the async receive_image function
    asyncio.run(async_receive_image())

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_video_stream():
    global receive_images
    while True:
        if receive_images is not None:
            _, jpeg = cv2.imencode('.jpg', receive_images)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@socketio.on('connect')
def handle_connect():
    print('클라이언트 연결됨!')
    # 연결되면 이미지를 바로 받을 수 있도록 한다
    socketio.start_background_task(receive_image)


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# Run the Flask app
if __name__ == '__main__':
    # Start the image receiving in a separate background task
    socketio.start_background_task(receive_image)  # Use Flask-SocketIO's background task function
    print("Starting Flask server...")
    socketio.run(app, host='0.0.0.0', port=8000)
