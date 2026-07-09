from ultralytics import YOLO
import cv2
import csv
import time
import os


print("Loading YOLO Pose Model...")

model = YOLO("yolo11n-pose.pt")

print("Model loaded!")


os.makedirs("data", exist_ok=True)

csv_file = "data/batting_pose_data.csv"


with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "time",

        "nose_x",
        "nose_y",

        "left_shoulder_x",
        "left_shoulder_y",
        "right_shoulder_x",
        "right_shoulder_y",

        "left_elbow_x",
        "left_elbow_y",
        "right_elbow_x",
        "right_elbow_y",

        "left_wrist_x",
        "left_wrist_y",
        "right_wrist_x",
        "right_wrist_y",

        "left_hip_x",
        "left_hip_y",
        "right_hip_x",
        "right_hip_y",

        "left_knee_x",
        "left_knee_y",
        "right_knee_x",
        "right_knee_y",

        "left_ankle_x",
        "left_ankle_y",
        "right_ankle_x",
        "right_ankle_y"
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


            current_time = round(
                time.time() - start_time,
                2
            )


            row = [
                current_time
            ]


            # COCO keypoint indexes
            important_points = [
                0,   # nose

                5,6, # shoulders

                7,8, # elbows

                9,10, # wrists

                11,12, # hips

                13,14, # knees

                15,16  # ankles
            ]


            for p in important_points:

                row.append(float(points[p][0]))
                row.append(float(points[p][1]))


            with open(csv_file, mode="a", newline="") as file:

                writer = csv.writer(file)
                writer.writerow(row)



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