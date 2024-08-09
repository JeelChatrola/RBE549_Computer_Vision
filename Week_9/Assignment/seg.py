import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
from collections import defaultdict

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

# Define colors for each class
colors = {
    'person': (0, 255, 0),
    'bicycle': (255, 0, 0),
    'car': (0, 0, 255)
}

def check_intersection(bbox, polygon):
    bbox_poly = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])
    return bbox_poly.intersects(polygon)

# Tracking history
track_history = defaultdict(lambda: [])
names = model.names
line_thickness = 2
track_thickness = 2
region_thickness = 2

def draw_tracks(frame, track, color):
    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=track_thickness)

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, conf=0.5, iou=0.5)

    # Draw the polygon
    cv2.polylines(frame, [np.array(polygon.exterior.coords, dtype=np.int32)], True, (255, 0, 255), 3)
    
    if results and len(results) > 0:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()
        
        for box, track_id, class_id, confidence in zip(boxes, track_ids, class_ids, confidences):
            x, y, w, h = box
            x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
            
            class_name = model.names[class_id]
            color = colors.get(class_name, (0, 255, 255))  # Default to yellow if class not in colors
            label = f'{class_name} ID: {track_id} Conf: {confidence:.2f}'
            
            bbox_center = (x1 + x2) / 2, (y1 + y2) / 2
            track = track_history[track_id]
            track.append((float(bbox_center[0]), float(bbox_center[1])))
            if len(track) > 30:
                track.pop(0)
            
            draw_tracks(frame, track, color)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if check_intersection((x1, y1, x2, y2), polygon):
                if class_name == 'person' and track_id not in humans_crossed:
                    human_count += 1
                    humans_crossed.add(track_id)
                    print(f"Person ID {track_id} crossed the line. Total humans: {human_count}")
                elif class_name == 'bicycle' and track_id not in bikes_crossed:
                    bike_count += 1
                    bikes_crossed.add(track_id)
                    print(f"Bicycle ID {track_id} crossed the line. Total bikes: {bike_count}")
                elif class_name == 'car' and track_id not in cars_crossed:
                    car_count += 1
                    cars_crossed.add(track_id)
                    print(f"Car ID {track_id} crossed the line. Total cars: {car_count}")

    cv2.putText(frame, f"Humans: {human_count}", (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Bikes: {bike_count}", (10, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Cars: {car_count}", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()