import math
from collections import deque



class BallTrackerUtils:
    """
    Utility functions for cricket ball analysis.

    Handles:
    - trajectory smoothing
    - ball speed calculation
    - movement distance
    """



    def __init__(self, buffer_size=10):

        self.positions = deque(
            maxlen=buffer_size
        )

        self.previous_position = None



    # ---------------------------------
    # Add new ball position
    # ---------------------------------

    def add_position(self, x, y):

        self.positions.append(
            (x, y)
        )



    # ---------------------------------
    # Smooth trajectory
    # ---------------------------------

    def get_smoothed_position(self):

        if len(self.positions) == 0:
            return None


        avg_x = sum(
            p[0]
            for p in self.positions
        ) / len(self.positions)


        avg_y = sum(
            p[1]
            for p in self.positions
        ) / len(self.positions)


        return (
            int(avg_x),
            int(avg_y)
        )



    # ---------------------------------
    # Distance between frames
    # ---------------------------------

    def calculate_distance(
            self,
            point1,
            point2
    ):

        dx = point2[0] - point1[0]

        dy = point2[1] - point1[1]


        return math.sqrt(
            dx ** 2 +
            dy ** 2
        )



    # ---------------------------------
    # Ball speed
    # ---------------------------------

    def calculate_speed(
            self,
            current_position,
            previous_position,
            time_difference
    ):

        if previous_position is None:
            return 0


        if time_difference <= 0:
            return 0


        distance = self.calculate_distance(
            previous_position,
            current_position
        )


        speed = distance / time_difference


        return speed



    # ---------------------------------
    # Trajectory points
    # ---------------------------------

    def get_trajectory(self):

        return list(
            self.positions
        )