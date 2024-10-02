# -*- coding: utf-8 -*-

import asyncio
import cv2
import websockets
import numpy as np
import socket
import json
from jetcam.csi_camera import CSICamera
from jetracer.nvidia_racecar import NvidiaRacecar

# jetracer 객체
car = NvidiaRacecar()

host_ip = socket.gethostbyname(socket.gethostname())
print("ASDASDSD", host_ip)

# 카메라 연결
camera = CSICamera(width=640, height=640)
camera.running = True
print("카메라 연결 완료")

async def stream_camera(websocket, path):

#     cap = cv2.VideoCapture(0)
#     cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! videoconvert ! video/x-raw,format=BGR ! appsink")

    while True:
#         ret, frame = cap.read()
        frame = camera.value

        _, img_encoded = cv2.imencode('.jpg', frame)
        data = img_encoded.tobytes()

        await websocket.send(data)
        # 보내는 속도 조절
        await asyncio.sleep(0.5)

    camera.release()

async def receive_angle(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        print(f"수신된 데이터: {data}")  # 수신된 데이터를 전체적으로 출력
        move_angle = data.get("move_angle")
        throttle = data.get("throttle", 0.3)  # 기본값 0
        print(f"수신된 throttle 값: {throttle}")
        if move_angle is not None:
            print("수신된 각도", move_angle)
            car.steering = move_angle
            car.throttle = throttle  # 속도 설정
            await asyncio.sleep(1.5)  # 대기 시간을 더 늘림
            car.throttle = 0
            car.steering = 0
        else:
            print("수신된 각도 없음")


    
async def start_server():
    # 웹소켓 서버
    server = await websockets.serve(stream_camera, '10.1.80.245', 5000)
    print("웹소켓 서버 시작")
    
    # 각도 수신 서버
    angle_server = await websockets.serve(receive_angle, '10.1.80.245', 5001)
    print("각도 수신 서버 시작")
    
    await server.wait_closed()

async def main():
    await start_server()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())