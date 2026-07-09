print("CricketVision AI starting...")

from ultralytics import YOLO
import cv2

print("Loading YOLO Pose model...")

# Load the YOLO pose model (downloads automatically the first time)
model = YOLO("yolo11n-pose.pt")

print("Model loaded!")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("CricketVision AI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()