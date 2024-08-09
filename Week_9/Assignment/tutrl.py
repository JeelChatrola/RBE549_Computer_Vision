import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, Point
from collections import defaultdict

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with the path to your YOLO model

# Load the video file using OpenCV
video_path = 'TrafficVideo.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Configuration
classes = [0, 1, 2]  # person, bicycle, car
names = model.names
line_thickness = 2
track_thickness = 2
region_thickness = 2

# Define counting regions (you can add more regions as needed)
counting_regions = [
    {
        "name": "Region 1",
        "polygon": Polygon([(550, 850), (1250, 600), (1700, 630), (1500, 950)]),
        "counts": defaultdict(int),
        "region_color": (255, 0, 255),
        "text_color": (255, 255, 255)
    }
]

# Tracking history
track_history = defaultdict(lambda: [])

# Helper functions
def draw_box_and_label(frame, box, label, color):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(frame, p1, p2, color, line_thickness)
    cv2.putText(frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, line_thickness)

def draw_tracks(frame, track, color):
    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=track_thickness)

def draw_regions(frame, regions):
    for region in regions:
        region_color = region["region_color"]
        polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
        cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)
        
        # Draw count for each class
        y_offset = 30
        for cls, count in region["counts"].items():
            label = f"{names[cls]}: {count}"
            cv2.putText(frame, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_color, line_thickness)
            y_offset += 30

# Main processing loop
vid_frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    vid_frame_count += 1

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, classes=classes)

    if results and len(results) > 0 and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            color = (int(hash(str(track_id)) % 256), int(hash(str(track_id * 2)) % 256), int(hash(str(track_id * 3)) % 256))
            label = f'{names[int(cls)]} ID: {track_id}'
            draw_box_and_label(frame, box, label, color)

            bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            track = track_history[track_id]
            track.append((float(bbox_center[0]), float(bbox_center[1])))
            if len(track) > 30:
                track.pop(0)
            draw_tracks(frame, track, color)

            # Check if detection inside regions
            for region in counting_regions:
                if region["polygon"].contains(Point(bbox_center)):
                    region["counts"][int(cls)] += 1

    # Draw regions and counts
    draw_regions(frame, counting_regions)

    # Display frame count
    cv2.putText(frame, f'Frame: {vid_frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show and write the frame
    cv2.imshow('frame', frame)
    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()