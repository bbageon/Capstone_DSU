# 필수 라이브러리 설치
pip install -r requirements.txt

# 코드 설명
해당 프로젝트는 NVIDIA Jetson Nano 기반의 Jetracer 플랫폼에서 자율주행 구현을 위한 딥러닝 알고리즘으로 장애물 회피 및 목표 탐색을 구현하였다.

jetracer.py => jetracer 플랫폼 내에서 작동하는 코드로 socket.io 를 통한 스트리밍 데이터 송신, 각도 수신으로 구성되어있음.

detection.py => YOLO v8 기반의 딥러닝 알고리즘으로, 충돌 영역 판별을 통한 장애물 회피 및 목표 영역 탐색을 구현하였음.