import asyncio
import cv2
import numpy as np
import websockets
from ultralytics import YOLO
import time
import math
import json
import sys

# Load the YOLOv8 model
model = YOLO('./runs/detect/train2/weights/last.pt')

# 모델 내의 클래스 받아오기
class_names = model.names

# 장애물과 목적지간의 각도를 계산하기 위한 기준점, 현재 로봇이 바라보고 있는 방향 width = 448 height = 448
center_x = 320
center_y = 640

# jetracer 에서 받아오는 이미지 크기
img_width = 640
img_height = 640

# 이미지 3등분하여 영역 설정
area_width = img_width // 3
# 각 영역의 범위 설정
area1_range = (0, area_width)
area2_range = (area_width, area_width * 2)
area3_range = (area_width * 2, img_width)

# jetracer 바퀴 각도를 위한 전처리 값, 바퀴값 [-1, 1] 사이, 최대각도 90도, 반대방향으로 가야하므로 - 값 
jetracer_value = -90

# 탐지된 객체가 어디에 있는지 확인하는 함수 
def determine_area(box_center_x):
    if area1_range[0] <= box_center_x < area1_range[1]:
        return 1
    elif area2_range[0] <= box_center_x < area2_range[1]:
        return 2
    elif area3_range[0] <= box_center_x < area3_range[1]:
        return 3
    else:
        return None

# 장애물과 목표의 영역에 따라 결정되는 Jetracer 가 어느 영역으로 가야하는지 결정하는 함수    
def determine_jetracer_area(goal_boundary, obstacle_boundary):
    # 목적지와 장애물이 모두 영역 1에 있는 경우
    if goal_boundary == 1:
        if obstacle_boundary == 1:
            # return 2
            return 0 # jetracer 각도 0 
        else:
            # return 1 
            return -0.5 # jetracer 각도 -45도
    # 목적지와 장애물이 모두 영역 2에 있는 경우
    elif goal_boundary == 2:
        if obstacle_boundary == 2:
            # return 1,3
            return -0.5 # jetracer 각도 45도
        else:
            # return 2 
            return 0 # jetracer 각도 0
    # 목적지와 장애물이 모두 영역 3에 있는 경우
    elif goal_boundary == 3:
        if obstacle_boundary == 3:
            # return 2
            return 0 # jetracer 각도 0
        else:
            # return 3
            return 0.5 # jetracer 각도 45도
    # 기타 경우
    else:
        return "오류 발생"


# 각도 계산 함수
def cal_rad(goal_location, obstacle_location):
    # 이미지 중심에서 각 점까지의 벡터 계산
    v1 = (goal_location[0] - center_x, goal_location[1] - center_y)
    v2 = (obstacle_location[0] - center_x, obstacle_location[1] - center_y)

    # 각 벡터의 크기 계산
    m1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    m2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    # 두 벡터의 내적 계산
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]

    # 코사인 법칙을 이용하여 각도 계산
    cos_angle = dot_product / (m1 * m2)

    # 각도를 라디안에서 도로 변환
    angle = math.acos(cos_angle)
    angle_degree = math.degrees(angle)
    
    # 벡터의 외적 계산
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]

    # 각도를 y축을 기준으로 왼쪽은 음수, 오른쪽은 양수값으로 변환
    if cross_product < 0:
        angle_degree = -angle_degree
    
    # jetracer 바퀴 각도값 [-1, 1] 사아
    jetracer_angle = round(angle_degree / jetracer_value, 2)

    return jetracer_angle

# 각도 전송 함수
async def send_angle(move_angle):
    uri = "ws://127.0.1.1:5001"
    async with websockets.connect(uri) as websocket:    
        # 각도 Jetracer 에 전송
        await websocket.send(json.dumps({"move_angle" : move_angle}))
        
# 현재 이미지 개수     
count = 0          
# 이미지 저장 함수                
async def save_image(img):
    global count
    # img_file_path = f"./received_image1_.jpg"
    img_file_path = f"./images/thirdimages/image{count}_.jpg"
    cv2.imwrite(img_file_path, img)
    count += 1
    print("Image saved as", img_file_path)
        
# 실시간 스트리밍 데이터 수신 함수
async def receive_image():
    global receive_images
    uri = "ws://127.0.1.1:5000"  # 서버의 주소 및 포트
    async with websockets.connect(uri) as websocket:
        
        while True:
            # 서버로부터 이미지 데이터 수신
            data = await websocket.recv()

            # 수신된 데이터를 이미지로 디코딩
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            receive_images = img

            # 이미지를 화면에 표시
            cv2.imshow("Received Image", img)
            cv2.waitKey(1)
            
            
            # await asyncio.sleep(0.1)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                await save_image(img)

# YOLO 모델 구동
async def detection_image():
    # 수신한 실시간 스트리밍 데이터 선언
    global receive_images
    results = model(receive_images)
        
    for result in results:
        boxes = result.boxes.xyxy

        # 검출된 객체가 모델에 있는지 확인
        labels = [class_names[int(label)] for label in result.boxes.cls]

        if len(boxes) > 0:

            # 목적지, 장애물
            goal_location = []
            obstacle_location = []

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box

                # bounding box 중심 좌표
                box_center_x = round((((x1 + x2) / 2)).item(), 5)
                box_center_y = round((((y1 + y2) / 2)).item(), 5)

                # 기준점과 bounding box 중심 거리 계산
                distance_to_center = round(np.sqrt((center_x - box_center_x) ** 2 + (center_y - box_center_y) ** 2), 2)
                

                if label == "goal":
                    goal_location = [box_center_x, box_center_y]
                else:
                    obstacle_location.append({"location": [box_center_x, box_center_y], "distance": distance_to_center})


                print("Bounding Box 중심 좌표:", (box_center_x, box_center_y))
                print("현재 어느 영역?", determine_area(box_center_x))
                print("감지된 객체 : ", label)
                print("거리 : ", distance_to_center)

            # goal 탐지 안 됐을 때 예외처리 
            if not goal_location:
                return print("goal 없음")
            
            # 장애물 감지된 경우에만 각도 계산
            if obstacle_location:
                
                move_angle = []
                area = []
                
                # 목적지가 어느 영역에 속하는지 확인
                goal_area = determine_area(goal_location[0])
                
                # 거리 기준으로 장애물 정렬
                obstacle_location.sort(key=lambda x: x["distance"])
                
                # 장애물 배열에 있는 좌표를 모두 계산
                for obstacle in obstacle_location:
                    
                    # 장애물 어디 영역인지 계산
                    obstacle_area = determine_area(obstacle["location"][0])
                    
                    # jetracer 어느 영역으로 가야하는지 계산
                    area.append(determine_jetracer_area(goal_area, obstacle_area))
                    
                    # 장애물과 목표의 라디안 각도를 jetracer 각도로 변환
                    move_angle.append(cal_rad(goal_location, obstacle["location"]))
                    
                print("각도 :", move_angle)
                print("현재 가야하는 영역 :", area)

                # # 각도 데이터를 서버로 전송
                # await send_angle(move_angle)
                
                # 영역 전송
                await send_angle(area[0])
            else:
                print("감지된 장애물이 없습니다.")
        else:
            print("감지된 객체 없음")

# Detection_task 를 5초마다 실행하기 위한 비동기 함수
async def run_detection():
    while True:
        await asyncio.sleep(5)  # 5초 대기
        # await detection_image()
        # if sys.stdin in asyncio.current_task().get_stack()[0].task.get_coros():
        #     # 키보드 입력 감지
        #     key = sys.stdin.read(1)   
        #     if key == 'd':
        #         print("탐지 작업 종료")
        #         break
            
async def main():
    receive_task = asyncio.create_task(receive_image())  # 이미지 수신을 비동기적으로 실행
    detection_task = asyncio.create_task(run_detection())  # main_processing()을 비동기적으로 실행
    await asyncio.gather(receive_task, detection_task)  # 두 작업을 병렬로 실행

# 비동기 함수 실행
asyncio.run(main())
