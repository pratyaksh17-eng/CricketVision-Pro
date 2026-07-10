from ultralytics import YOLO
import cv2
import csv
import os
import time

from utils.video_processor import VideoProcessor
from utils.pose_analyzer import PoseAnalyzer
from utils.visualization import draw_dashboard


# -------------------------------
# Paths
# -------------------------------

INPUT_VIDEO = "input_videos/batting.mp4"
OUTPUT_VIDEO = "output_videos/batting_analysis.mp4"
CSV_FILE = "data/batting_pose_data.csv"


# -------------------------------
# Create folders
# -------------------------------

os.makedirs("data", exist_ok=True)
os.makedirs("output_videos", exist_ok=True)


# -------------------------------
# Load model
# -------------------------------

print("Loading YOLO Pose Model...")

model = YOLO("yolo11n-pose.pt")

print("Model loaded!")


# -------------------------------
# Video Processor
# -------------------------------

video = VideoProcessor(
    INPUT_VIDEO,
    OUTPUT_VIDEO
)

analyzer = PoseAnalyzer()


# -------------------------------
# CSV
# -------------------------------

with open(CSV_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "frame",
        "left_elbow",
        "right_elbow",
        "swing_speed",
        "head_stability",
        "shoulder_rotation",
        "hip_rotation",
        "swing_phase"
    ])


print("Starting analysis...")


# -------------------------------
# Main Loop
# -------------------------------

while True:

    success, frame = video.read()

    if not success:
        break

    results = model(frame, verbose=False)

    annotated = results[0].plot()

    if (
        results[0].keypoints is not None
        and
        len(results[0].keypoints.xy) > 0
    ):

        points = results[0].keypoints.xy[0]

        metrics = analyzer.analyze(points)
        # -------------------------------
        # Save CSV
        # -------------------------------

        with open(CSV_FILE, "a", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                video.current_frame,
                round(metrics["left_angle"], 2),
                round(metrics["right_angle"], 2),
                round(metrics["swing_speed"], 2),
                round(metrics["head_stability"], 2),
                round(metrics["shoulder_rotation"], 2),
                round(metrics["hip_rotation"], 2),
                metrics["swing_phase"]
            ])

        # -------------------------------
        # Draw Dashboard
        # -------------------------------

        annotated = draw_dashboard(
            annotated,
            metrics["swing_phase"],
            metrics["swing_speed"],
            metrics["left_angle"],
            metrics["right_angle"],
            metrics["head_stability"],
            metrics["shoulder_rotation"],
            metrics["hip_rotation"]
        )

    # -------------------------------
    # Progress
    # -------------------------------

    progress = video.progress()

    cv2.putText(
        annotated,
        f"Processing: {progress:.1f}%",
        (20, annotated.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "CricketVision-Pro Video Analysis",
        annotated
    )

    video.write(annotated)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break


# -------------------------------
# Cleanup
# -------------------------------

video.release()

print("\nAnalysis Complete!")
print(f"Output video : {OUTPUT_VIDEO}")
print(f"CSV saved    : {CSV_FILE}")