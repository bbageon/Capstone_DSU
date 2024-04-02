from PIL import Image
import requests
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Define a function to extract image center and bounding box coordinates
def extract_coordinates(image_url):
    # Predict on the image
    results = model(image_url)

    # Load the image using PIL
    image = Image.open(requests.get(image_url, stream=True).raw)
    image_width, image_height = image.size

    # Extract image center coordinates
    image_center = (image_width / 2, image_height / 2)

    # Extract bounding box coordinates
    for label, confidence, box in results.xyxy[0]:
        # Convert box coordinates from normalized to absolute values
        xmin, ymin, xmax, ymax = box
        xmin *= image_width
        xmax *= image_width
        ymin *= image_height
        ymax *= image_height

        # Calculate bounding box center coordinates
        box_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)

        print("Label:", label)
        print("Confidence:", confidence)
        print("Bounding Box Center:", box_center)
        print("Bounding Box Coordinates:", (xmin, ymin, xmax, ymax))
        print()

# Test the function with an example image URL
image_url = "https://ultralytics.com/images/bus.jpg"
extract_coordinates(image_url)
