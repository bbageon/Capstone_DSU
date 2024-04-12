import asyncio
import cv2
import numpy as np
import websockets
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# 이미지 중심 좌표 (0,0) width = 224 height = 224
img_center_x = 112
img_center_y = 112

# 모델 내의 클래스 받아오기
class_names = model.names

async def receive_image():
    uri = "ws://10.1.169.172:5000"  # 서버의 주소 및 포트
    async with websockets.connect(uri) as websocket:
        
        # 속도 조절
        time.sleep(2)
        while True:
            # 서버로부터 이미지 데이터 수신
            data = await websocket.recv()

            # 수신된 데이터를 이미지로 디코딩
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 이미지를 화면에 표시
            cv2.imshow("Received Image", img)
            cv2.waitKey(1)
            
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
                    for box, label in zip(boxes, labels):
                        x1, y1, x2, y2 = box
                        
                        # bounding box 중심 좌표
                        box_center_x = round((((x1 + x2) / 2) / 224).item(), 6)
                        box_center_y = round((((y1 + y2) / 2) / 224).item(), 6)
                        # box_center_x = (box[0] + box[2]) / 2
                        # box_center_y = (box[1] + box[3]) / 2
                        
                        # 이미지 중심과 bounding box 중심 거리 계산
                        distance_to_center = np.sqrt((img_center_x - box_center_x) ** 2 + (img_center_y - box_center_y) ** 2)
                        
                        print("Bounding Box 중심 좌표:", (box_center_x, box_center_y))
                        print("이미지 중심과의 거리:", distance_to_center)
                        print("감지된 객체 : ", label)
                else:
                    print("감지된 객체 없음")
                
                
            # cv2.destroyAllWindows()
            
            # 이미지를 로컬 디렉토리에 저장
            # filename = "received_image.jpg"  # 저장할 파일명
            # cv2.imwrite(filename, img)
            # print(f"이미지가 {filename}으로 저장되었습니다.")

async def main():
    await receive_image()

if __name__ == "__main__":
    asyncio.run(main())