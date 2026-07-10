import math
import time

from utils.metrics import (
    calculate_angle,
    calculate_head_stability,
    calculate_shoulder_rotation,
    calculate_hip_rotation
)

from utils.swing_phase import SwingPhaseDetector


class PoseAnalyzer:

    def __init__(self):

        self.previous_right_wrist = None
        self.previous_head = None
        self.previous_frame_time = time.time()
        self.swing_speed = 0

        self.swing_detector = SwingPhaseDetector()

    def analyze(self, points):

        nose = tuple(points[0])

        left_shoulder = tuple(points[5])
        right_shoulder = tuple(points[6])

        left_elbow = tuple(points[7])
        right_elbow = tuple(points[8])

        left_wrist = tuple(points[9])
        right_wrist = tuple(points[10])

        left_hip = tuple(points[11])
        right_hip = tuple(points[12])

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

        current_time = time.time()

        if self.previous_right_wrist is not None:

            dx = right_wrist[0] - self.previous_right_wrist[0]
            dy = right_wrist[1] - self.previous_right_wrist[1]

            distance = math.sqrt(dx * dx + dy * dy)

            dt = current_time - self.previous_frame_time

            if dt > 0:

                instant_speed = distance / dt

                self.swing_speed = (
                    self.swing_speed * 0.8
                    + instant_speed * 0.2
                )

        self.previous_right_wrist = right_wrist
        self.previous_frame_time = current_time

        shoulder_rotation = calculate_shoulder_rotation(
            left_shoulder,
            right_shoulder
        )

        hip_rotation = calculate_hip_rotation(
            left_hip,
            right_hip
        )

        head_stability = calculate_head_stability(
            self.previous_head,
            nose
        )

        self.previous_head = nose

        wrist_velocity = self.swing_detector.calculate_velocity(
            right_wrist
        )

        swing_phase = self.swing_detector.detect_phase(
            wrist_velocity
        )

        return {
            "left_angle": left_angle,
            "right_angle": right_angle,
            "swing_speed": self.swing_speed,
            "head_stability": head_stability,
            "shoulder_rotation": shoulder_rotation,
            "hip_rotation": hip_rotation,
            "swing_phase": swing_phase
        }