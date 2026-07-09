import cv2
import csv
import os
import time

from utils.ball_utils import BallTrackerUtils
from utils.ball_detector import BallDetector



print("Starting CricketVision Ball Tracker...")



# ---------------------------------
# File paths
# ---------------------------------

input_video = "input_videos/cricket.mp4"

output_video = "output_videos/ball_tracking_output.mp4"

csv_file = "data/ball_tracking.csv"

model_path = "models/cricket_ball_model.pt"



# ---------------------------------
# Create folders
# ---------------------------------

os.makedirs(
    "output_videos",
    exist_ok=True
)

os.makedirs(
    "data",
    exist_ok=True
)



# ---------------------------------
# Load Ball Detector
# ---------------------------------

print("Initializing Ball Detector...")

ball_detector = BallDetector(
    model_path
)



# ---------------------------------
# Open video
# ---------------------------------

cap = cv2.VideoCapture(
    input_video
)


if not cap.isOpened():

    print("Cannot open video:")
    print(input_video)
    exit()



width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)


height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


fps = cap.get(
    cv2.CAP_PROP_FPS
)



# ---------------------------------
# Output video
# ---------------------------------

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)



# ---------------------------------
# CSV
# ---------------------------------

with open(
    csv_file,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "time",
            "ball_x",
            "ball_y",
            "speed"
        ]
    )



# ---------------------------------
# Tracking system
# ---------------------------------

tracker = BallTrackerUtils(
    buffer_size=15
)


previous_position = None

previous_time = time.time()


start_time = time.time()


frame_count = 0



# ---------------------------------
# Main loop
# ---------------------------------

while True:


    success, frame = cap.read()


    if not success:
        break



    frame_count += 1



    current_time = time.time()



    # ---------------------------------
    # Detect ball
    # ---------------------------------

    ball_position = ball_detector.detect_ball(
        frame
    )



    speed = 0



    if ball_position is not None:


        x, y = ball_position



        tracker.add_position(
            x,
            y
        )



        smooth_position = (
            tracker.get_smoothed_position()
        )



        speed = tracker.calculate_speed(
            smooth_position,
            previous_position,
            current_time - previous_time
        )



        previous_position = smooth_position

        previous_time = current_time



        elapsed = round(
            current_time - start_time,
            3
        )



        with open(
            csv_file,
            "a",
            newline=""
        ) as file:


            writer = csv.writer(file)


            writer.writerow(
                [
                    elapsed,
                    smooth_position[0],
                    smooth_position[1],
                    speed
                ]
            )



        cv2.circle(
            frame,
            smooth_position,
            8,
            (0,255,255),
            -1
        )



    # ---------------------------------
    # Draw trajectory
    # ---------------------------------

    trajectory = tracker.get_trajectory()



    for point in trajectory:

        cv2.circle(
            frame,
            point,
            4,
            (0,255,255),
            -1
        )



    # ---------------------------------
    # Display information
    # ---------------------------------

    cv2.putText(
        frame,
        "CricketVision Ball Tracker",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,255),
        2
    )


    cv2.putText(
        frame,
        f"Frames: {frame_count}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )


    cv2.putText(
        frame,
        f"Ball Speed: {speed:.2f}px/s",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )



    out.write(frame)



    cv2.imshow(
        "CricketVision Ball Tracker",
        frame
    )



    if cv2.waitKey(1) & 0xFF == ord("q"):

        break



cap.release()

out.release()

cv2.destroyAllWindows()



print("Ball tracking complete!")
print("Saved video:", output_video)
print("Saved data:", csv_file)