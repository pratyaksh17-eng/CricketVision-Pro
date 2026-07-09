import math


class SwingPhaseDetector:

    def __init__(self):
        self.previous_wrist = None
        self.phase = "READY"


    def calculate_velocity(self, current_wrist):
        """
        Calculates wrist movement speed.
        """

        if self.previous_wrist is None:
            self.previous_wrist = current_wrist
            return 0


        dx = (
            current_wrist[0] -
            self.previous_wrist[0]
        )

        dy = (
            current_wrist[1] -
            self.previous_wrist[1]
        )


        distance = math.sqrt(
            dx ** 2 +
            dy ** 2
        )


        self.previous_wrist = current_wrist

        return distance



    def detect_phase(self, wrist_speed):
        """
        Determines batting phase based on wrist movement.
        """

        if wrist_speed < 5:
            self.phase = "READY"


        elif wrist_speed < 25:
            self.phase = "BACKSWING"


        elif wrist_speed < 60:
            self.phase = "DOWNSWING"


        elif wrist_speed >= 60:
            self.phase = "IMPACT"


        return self.phase