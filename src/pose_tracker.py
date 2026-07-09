from ultralytics import YOLO
import cv2
import csv
import time
import os
import math


def calculate_angle(a, b, c):
    """Returns the angle ABC in degrees."""

    ax, ay = a
    bx, by = b
    cx, cy = c

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    dot = ba[0] * bc[0] + ba[1] * bc[1]

    mag1 = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag2 = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag1 == 0 or mag2 == 0:
        return 0

    cosine = dot / (mag1 * mag2)
    cosine = max(-1, min(1, cosine))

    return math.degrees(math.acos(cosine))


print("Loading YOLO Pose Model...")

model = YOLO("yolo11n-pose.pt")

print("Model loaded!")

os.makedirs("data", exist_ok=True)

csv_file = "data/batting_pose_data.csv"

with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "time",
        "nose_x", "nose_y",
        "left_shoulder_x", "left_shoulder_y",
        "right_shoulder_x", "right_shoulder_y",
        "left_elbow_x", "left_elbow_y",
        "right_elbow_x", "right_elbow_y",
        "left_wrist_x", "left_wrist_y",
        "right_wrist_x", "right_wrist_y",
        "left_hip_x", "left_hip_y",
        "right_hip_x", "right_hip_y",
        "left_knee_x", "left_knee_y",
        "right_knee_x", "right_knee_y",
        "left_ankle_x", "left_ankle_y",
        "right_ankle_x", "right_ankle_y"
    ])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access webcam")
    exit()

start_time = time.time()

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    if results[0].keypoints is not None:

        keypoints = results[0].keypoints.xy

        if len(keypoints) > 0:

            points = keypoints[0]

            # -------- ANGLE CALCULATIONS -------- #

            left_shoulder = tuple(points[5])
            left_elbow = tuple(points[7])
            left_wrist = tuple(points[9])

            right_shoulder = tuple(points[6])
            right_elbow = tuple(points[8])
            right_wrist = tuple(points[10])

            left_angle = calculate_angle(
                left_shoulder,
                left_elbow,
                left_wrist
            )

            right_angle = calculate_angle(
                right_shoulder,
                right_elbow,
                right_wrist
            )

            # -------- CSV DATA -------- #

            current_time = round(
                time.time() - start_time,
                2
            )

            row = [current_time]

            important_points = [
                0,
                5, 6,
                7, 8,
                9, 10,
                11, 12,
                13, 14,
                15, 16
            ]

            for p in important_points:
                row.append(float(points[p][0]))
                row.append(float(points[p][1]))

            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)

            # -------- DISPLAY ANGLES -------- #

            cv2.putText(
                annotated_frame,
                f"Left Elbow: {left_angle:.1f} deg",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                f"Right Elbow: {right_angle:.1f} deg",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow(
        "CricketVision AI - Batting Data Collection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Batting data collection complete!")
print("Saved to:", csv_file)