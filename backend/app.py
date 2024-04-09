from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np
import cv2
import base64
import socket

app = Flask(__name__)
socketio = SocketIO(app)
# IP 주소
host_ip = socket.gethostbyname(socket.gethostname())

@app.route('/')
def index():
    print("페이지 접속완료")
    return render_template('index.html')

@socketio.on('image')
def handle_image(data):
    # base64로 인코딩된 이미지 데이터 디코딩
    image_data = base64.b64decode(data.split(',')[1])
    nparr = np.fromstring(image_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 이미지 처리 등을 수행할 수 있음
    # 예시: 이미지를 그대로 웹 페이지에 전송
    _, img_encoded = cv2.imencode('.jpg', img_np)
    img_as_text = base64.b64encode(img_encoded).decode('utf-8')
    socketio.emit('image_processed', img_as_text)

if __name__ == '__main__':
    print("서버 실행 완료")
    print("IP주소 :", host_ip)
    socketio.run(app, host='192.168.116.165', port=5000, debug=True)
