from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Perform inference on the image
results = model("https://ultralytics.com/images/bus.jpg")

# Extract bounding boxes, classes, names, and confidences
boxes = results[0].boxes.xyxy.tolist()
classes = results[0].boxes.cls.tolist()
names = results[0].names
confidences = results[0].boxes.conf.tolist()
# Bounding Box 중앙값
center = []

# Iterate through the results
for box, cls, conf in zip(boxes, classes, confidences):
    x1, y1, x2, y2 = box
    confidence = conf
    detected_class = cls
    name = names[int(cls)]
    # Calculate the center of the bounding box
    center.append(((x1 + x2) / 2, (y1 + y2) / 2))
    
# print("boxes : ", boxes)
# print("classes : ", classes)
# print("names : ", names)
# print("confidences : ", confidences)
print("center :", center)

