import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, LineString

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with the path to your YOLO model

# Load the video file using OpenCV
video_path = 'TrafficVideo.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the sidewalk segmentation lines
line1 = [(550, 850), (1250, 600)]
line2 = [(1500, 950), (1700, 630)]
line3 = [(1250, 600), (1700, 630)]
line4 = [(550, 850), (1500, 950)]

# Create a polygon from the lines
polygon = Polygon([line1[0], line1[1], line2[1], line2[0]])

human_count = 0
bike_count = 0
car_count = 0
humans_crossed = set()
bikes_crossed = set()
cars_crossed = set()

def check_intersection(bbox, polygon):
    bbox_poly = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])
    return bbox_poly.intersects(polygon)

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get detections from your model
    results = model.track(frame, persist=True)

    # Prepare detections
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            obj_id = int(box.id[0].item()) if box.id is not None else -1
            detections.append([x1, y1, x2, y2, conf, cls, obj_id])

    for detection in detections:
        x1, y1, x2, y2, conf, cls, obj_id = detection
        label = f'{model.names[int(cls)]} ID: {obj_id} {conf:.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if check_intersection((x1, y1, x2, y2), polygon):
            if model.names[int(cls)] == 'person' and obj_id not in humans_crossed:
                human_count += 1
                humans_crossed.add(obj_id)
                print(f"Person ID {obj_id} crossed the line. Total humans: {human_count}")
            elif model.names[int(cls)] == 'bicycle' and obj_id not in bikes_crossed:
                bike_count += 1
                bikes_crossed.add(obj_id)
                print(f"Bicycle ID {obj_id} crossed the line. Total bikes: {bike_count}")
            elif model.names[int(cls)] == 'car' and obj_id not in cars_crossed:
                car_count += 1
                cars_crossed.add(obj_id)
                print(f"Car ID {obj_id} crossed the line. Total cars: {car_count}")

    cv2.putText(frame, f"Humans: {human_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"Bikes: {bike_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"Cars: {car_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw the polygon
    cv2.polylines(frame, [np.array(polygon.exterior.coords, dtype=np.int32)], True, (255, 0, 255), 3)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()