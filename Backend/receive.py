import asyncio
import cv2
import numpy as np
import websockets
from ultralytics import YOLO
import time
import math
import json

# Load the YOLOv8 model
model = YOLO('./yolov8n.pt')

# 모델 내의 클래스 받아오기
class_names = model.names

# 이미지 중심 좌표 (0,0) width = 224 height = 224
img_center_x = 112
img_center_y = 112

# 각도 계산 함수
def cal_rad(arr1, arr2):
    # 이미지 중심에서 각 점까지의 벡터 계산
    vector1 = (arr1[0] - img_center_x, arr1[1] - img_center_y)
    vector2 = (arr2[0] - img_center_x, arr2[1] - img_center_y)

    # 각 벡터의 크기 계산
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # 두 벡터의 내적 계산
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # 코사인 법칙을 이용하여 각도 계산
    cos_angle = dot_product / (magnitude1 * magnitude2)

    # 각도를 라디안에서 도로 변환
    angle = math.acos(cos_angle)
    angle_deg = math.degrees(angle)
    
    # 벡터의 외적 계산
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]

    # 각도를 y축을 기준으로 왼쪽은 음수, 오른쪽은 양수값으로 변환
    if cross_product < 0:
        angle_deg = -angle_deg
    
    return angle_deg

# 각도 전송 함수
async def send_angle(move_angle):
    uri = "ws://10.1.169.172:5001"
    async with websockets.connect(uri) as websocket:    
        # 각도 Jetracer 에 전송
        await websocket.send(json.dumps({"move_angle" : move_angle}))

# yolo 모델 구동 및 각도 계산 및 전송 함수
async def detection_obstacle(img):
     # YOLO 모델 구동
        results = model(img, stream=True)
        for result in results:
            boxes = result.boxes.xyxy  # Boxes object for bounding box outputs
            
            # 검출된 객체가 모델에 있는지 확인
            labels = [class_names[int(label)] for label in result.boxes.cls]
            
            # masks = result.masks  # Masks object for segmentation masks outputs
            # keypoints = result.keypoints  # Keypoints object for pose outputs
            # probs = result.probs  # Probs object for classification outputs
            # result.show()  # display to screen
            # result.save(filename='result.jpg')
        
            if len(boxes) > 0: 
                
                # 목적지, 장애물, 각도
                goal_location = []
                obstacle_location = []
                move_angle = []
                
                # # 속도 조절
                # time.sleep(2)
                
                for box, label in zip(boxes, labels):
                    x1, y1, x2, y2 = box
                    print("x1 :", x1)
                    print("x2 :", x2)
                    print("y1 :", y1)
                    print("y2 :", y2)
                    
                    # bounding box 중심 좌표
                    box_center_x = round((((x1 + x2) / 2) / 224).item(), 1)
                    box_center_y = round((((y1 + y2) / 2) / 224).item(), 1)
                    # box_center_x = (box[0] + box[2]) / 2
                    # box_center_y = (box[1] + box[3]) / 2
                    
                    # 이미지 중심과 bounding box 중심 거리 계산
                    distance_to_center = np.sqrt((img_center_x - box_center_x) ** 2 + (img_center_y - box_center_y) ** 2)
                    
                    if label == "goal":
                        goal_location.append([box_center_x, box_center_y])
                    else:
                        obstacle_location.append([box_center_x, box_center_y])
                    
                    print("Bounding Box 중심 좌표:", (box_center_x, box_center_y))
                    # print("이미지 중심과의 거리:", distance_to_center)
                    print("감지된 객체 : ", label)
                
                # 각도계산해서 추가
                goal_location.append([112,224])
                
                # 장애물 감지된 경우에만 각도 계산
                if obstacle_location:
                    # 장애물 배열에 있는 좌표를 모두 계산
                    for i in range(len(obstacle_location)):
                        move_angle.append(cal_rad(goal_location[0], obstacle_location[i]))
                    print("각도 :", move_angle)
                    
                    # 각도 데이터를 서버로 전송
                    send_angle(move_angle)
                else:
                    print("감지된 장애물이 없습니다.")
                    
                # 각도 전송
                await send_angle(move_angle)
                # YOLO predict 10초마다 한번씩 실행
                await asyncio.sleep(10)
            else:
                print("감지된 객체 없음") 

# 현재 이미지 개수     
count = 0          
# 이미지 저장 함수                
async def save_image(img):
    global count
    img_file_path = f"./received_image1_.jpg"
    # img_file_path = f"./images/secondimageset/image{count}_.jpg"
    cv2.imwrite(img_file_path, img)
    count += 1
    print("Image saved as", img_file_path)

# 실시간 스트리밍 데이터 수신 함수
async def receive_image():
    uri = "ws://10.1.169.172:5000"  # 서버의 주소 및 포트
    async with websockets.connect(uri) as websocket:
        
        while True:
            # 서버로부터 이미지 데이터 수신
            data = await websocket.recv()

            # 수신된 데이터를 이미지로 디코딩
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 이미지를 화면에 표시
            cv2.imshow("Received Image", img)
            cv2.waitKey(1)
            
            # await detection_obstacle(img)
            
            # await asyncio.sleep(0.1)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                await save_image(img)

async def main():
    await receive_image()

if __name__ == "__main__":
    asyncio.run(main())