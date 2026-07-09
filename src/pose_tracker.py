from ultralytics import YOLO
import cv2
import csv
import time
import os
import math

from utils.metrics import (
    calculate_angle,
    calculate_shoulder_rotation,
    calculate_hip_rotation,
    calculate_head_stability
)

from utils.swing_phase import SwingPhaseDetector

from utils.visualization import draw_dashboard


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
        "right_hip_x", "right_hip_y"
    ])



cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("Cannot access webcam")
    exit()



start_time = time.time()



# Tracking variables

previous_right_wrist = None
previous_head = None

previous_frame_time = time.time()

swing_speed = 0
fps = 0

head_stability = 100

shoulder_rotation = 0
hip_rotation = 0

swing_phase = "READY"


swing_detector = SwingPhaseDetector()



while True:


    frame_start = time.time()


    success, frame = cap.read()


    if not success:
        break



    results = model(frame, verbose=False)


    annotated_frame = results[0].plot()



    if results[0].keypoints is not None:


        keypoints = results[0].keypoints.xy


        if len(keypoints) > 0:


            points = keypoints[0]



            # -------------------------------
            # Joint positions
            # -------------------------------

            nose = tuple(points[0])


            left_shoulder = tuple(points[5])
            right_shoulder = tuple(points[6])


            left_elbow = tuple(points[7])
            right_elbow = tuple(points[8])


            left_wrist = tuple(points[9])
            right_wrist = tuple(points[10])


            left_hip = tuple(points[11])
            right_hip = tuple(points[12])



            # -------------------------------
            # Angles
            # -------------------------------

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



            # -------------------------------
            # Swing speed
            # -------------------------------

            current_time = time.time()


            if previous_right_wrist is not None:


                dx = (
                    right_wrist[0]
                    -
                    previous_right_wrist[0]
                )


                dy = (
                    right_wrist[1]
                    -
                    previous_right_wrist[1]
                )


                distance = math.sqrt(
                    dx ** 2 +
                    dy ** 2
                )


                dt = (
                    current_time
                    -
                    previous_frame_time
                )


                if dt > 0:


                    instant_speed = distance / dt


                    swing_speed = (
                        swing_speed * 0.8
                        +
                        instant_speed * 0.2
                    )



            previous_right_wrist = right_wrist
            previous_frame_time = current_time



            # -------------------------------
            # Advanced biomechanics
            # -------------------------------


            shoulder_rotation = calculate_shoulder_rotation(
                left_shoulder,
                right_shoulder
            )


            hip_rotation = calculate_hip_rotation(
                left_hip,
                right_hip
            )



            head_stability = calculate_head_stability(
                previous_head,
                nose
            )


            previous_head = nose



            # -------------------------------
            # Swing phase
            # -------------------------------

            wrist_velocity = swing_detector.calculate_velocity(
                right_wrist
            )


            swing_phase = swing_detector.detect_phase(
                wrist_velocity
            )



            # -------------------------------
            # Save CSV
            # -------------------------------

            elapsed = round(
                time.time() - start_time,
                2
            )


            row = [
                elapsed
            ]


            for p in [
                0,
                5,6,
                7,8,
                9,10,
                11,12
            ]:


                row.append(
                    float(points[p][0])
                )

                row.append(
                    float(points[p][1])
                )



            with open(
                csv_file,
                mode="a",
                newline=""
            ) as file:


                writer = csv.writer(file)

                writer.writerow(row)



            # -------------------------------
            # Dashboard
            # -------------------------------


            annotated_frame = draw_dashboard(
                annotated_frame,
                swing_phase,
                swing_speed,
                left_angle,
                right_angle,
                head_stability,
                shoulder_rotation,
                hip_rotation
            )



    # FPS

    frame_end = time.time()


    fps = 1 / (
        frame_end - frame_start
    )


    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )



    cv2.imshow(
        "CricketVision AI - Batting Analyzer",
        annotated_frame
    )



    if cv2.waitKey(1) & 0xFF == ord("q"):

        break



cap.release()

cv2.destroyAllWindows()


print("Batting analysis complete!")
print("Saved to:", csv_file)