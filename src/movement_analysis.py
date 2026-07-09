import pandas as pd
import matplotlib.pyplot as plt
import os


print("Loading CricketVision AI dataset...")


# Load data
file_path = "data/pose_data.csv"

df = pd.read_csv(file_path)


print("\nDataset Preview:")
print(df.head())


print("\nDataset Information:")
print(df.info())


# Calculate wrist movement distance

df["wrist_distance"] = (
    ((df["left_wrist_x"].diff())**2 +
     (df["left_wrist_y"].diff())**2) ** 0.5
)


# Remove first empty value

df = df.dropna()


# Statistics

average_movement = df["wrist_distance"].mean()

maximum_movement = df["wrist_distance"].max()


print("\n--- CricketVision AI Report ---")

print(
    f"Average Wrist Movement: {average_movement:.2f} pixels/frame"
)

print(
    f"Maximum Wrist Movement: {maximum_movement:.2f} pixels/frame"
)


# Create graph

plt.figure(figsize=(10,5))

plt.plot(
    df["time"],
    df["wrist_distance"]
)

plt.xlabel("Time (seconds)")
plt.ylabel("Wrist Movement")
plt.title("CricketVision AI - Wrist Movement Analysis")


# Save graph

os.makedirs("output", exist_ok=True)

plt.savefig(
    "output/wrist_movement.png"
)

plt.show()


print("\nAnalysis complete!")
print("Graph saved in output/wrist_movement.png")