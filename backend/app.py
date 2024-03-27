from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

# 웹소켓 이벤트 핸들러
@socketio.on('image')
def handle_image(image):
    # 클라이언트로부터 받은 이미지 데이터 처리
    # 이 예시에서는 클라이언트로부터 받은 이미지를 다시 클라이언트에게 전송하여 화면에 표시합니다.
    emit('stream', image, broadcast=True)

# 기본 경로
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # 서버 실행
    socketio.run(app, host='0.0.0.0', port=5000)
