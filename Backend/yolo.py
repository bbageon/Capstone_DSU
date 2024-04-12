from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Perform inference on the image
results = model("https://ultralytics.com/images/bus.jpg")

# Get the class names used by the model
class_names = model.names
print(class_names, "@#@#@#")

for result in results:
    boxes = result.boxes.xyxy  # Boxes object for bounding box outputs
    labels = result.boxes.cls  # Class indexes of detected objects
    
    # Convert tensor labels to integers
    labels = [int(label) for label in labels]
    print("labels", labels)

    # Convert class indexes to class names
    class_names_detected = [class_names[i] for i in labels]
    print("검출된 객체의 클래스 이름:", class_names_detected)
    print("Bounding Boxes:", boxes)
