from ultralytics import YOLO
import cv2
import csv
import time
import os


print("Loading YOLO Pose Model...")

model = YOLO("yolo11n-pose.pt")

print("Model ready!")


os.makedirs("data", exist_ok=True)

csv_file = "data/batting_session.csv"


recording = False


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera error")
    exit()


print("\nCricketVision AI Batting Mode")
print("Press SPACE to start recording")
print("Press Q to stop\n")


start_time = None


while True:

    success, frame = cap.read()

    if not success:
        break


    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()


    # Start recording
    key = cv2.waitKey(1) & 0xFF


    if key == ord(" "):

        recording = True
        start_time = time.time()

        print("Recording started!")

        with open(csv_file, "w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                "time",
                "nose_x","nose_y",
                "left_shoulder_x","left_shoulder_y",
                "right_shoulder_x","right_shoulder_y",
                "left_elbow_x","left_elbow_y",
                "right_elbow_x","right_elbow_y",
                "left_wrist_x","left_wrist_y",
                "right_wrist_x","right_wrist_y",
                "left_hip_x","left_hip_y",
                "right_hip_x","right_hip_y"
            ])



    if recording:

        if results[0].keypoints is not None:

            keypoints = results[0].keypoints.xy


            if len(keypoints) > 0:

                points = keypoints[0]


                row = [
                    round(time.time()-start_time,2)
                ]


                selected = [
                    0,
                    5,6,
                    7,8,
                    9,10,
                    11,12
                ]


                for p in selected:

                    row.append(float(points[p][0]))
                    row.append(float(points[p][1]))


                with open(csv_file,"a",newline="") as file:

                    writer = csv.writer(file)

                    writer.writerow(row)



    if recording:

        cv2.putText(
            annotated_frame,
            "RECORDING",
            (30,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )


    cv2.imshow(
        "CricketVision AI - Batting Mode",
        annotated_frame
    )


    if key == ord("q"):
        break



cap.release()
cv2.destroyAllWindows()


if recording:
    print("\nSession saved!")
    print(csv_file)

else:
    print("\nNo session recorded.")