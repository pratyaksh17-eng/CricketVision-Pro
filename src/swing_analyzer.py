import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print("Loading batting data...")


df = pd.read_csv("data/batting_pose_data.csv")


print("Frames analyzed:", len(df))


# Calculate wrist movement

df["right_wrist_speed"] = np.sqrt(
    df["right_wrist_x"].diff()**2 +
    df["right_wrist_y"].diff()**2
)


df["left_wrist_speed"] = np.sqrt(
    df["left_wrist_x"].diff()**2 +
    df["left_wrist_y"].diff()**2
)


df = df.dropna()


# Find fastest movement

max_speed = max(
    df["right_wrist_speed"].max(),
    df["left_wrist_speed"].max()
)


average_speed = (
    df["right_wrist_speed"].mean()
)


# Head movement

head_movement = np.sqrt(
    df["nose_x"].diff()**2 +
    df["nose_y"].diff()**2
)


average_head_movement = head_movement.mean()


print("\n------ CricketVision AI Report ------")

print(
    f"Average Wrist Speed: {average_speed:.2f} pixels/frame"
)

print(
    f"Maximum Swing Speed: {max_speed:.2f} pixels/frame"
)

print(
    f"Head Movement Score: {average_head_movement:.2f}"
)


# Plot swing speed

plt.figure(figsize=(10,5))

plt.plot(
    df["time"],
    df["right_wrist_speed"]
)

plt.xlabel("Time")
plt.ylabel("Wrist Speed")

plt.title(
    "CricketVision AI - Swing Speed Analysis"
)


plt.savefig(
    "output/swing_speed.png"
)


plt.show()


print("\nSwing analysis complete!")
print("Saved graph: output/swing_speed.png")