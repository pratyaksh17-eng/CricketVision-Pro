import math


def calculate_angle(a, b, c):
    """
    Calculates angle ABC in degrees.

    a = first point
    b = middle joint
    c = last point
    """

    ax, ay = a
    bx, by = b
    cx, cy = c

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    dot_product = (
        ba[0] * bc[0] +
        ba[1] * bc[1]
    )

    magnitude_ba = math.sqrt(
        ba[0] ** 2 +
        ba[1] ** 2
    )

    magnitude_bc = math.sqrt(
        bc[0] ** 2 +
        bc[1] ** 2
    )

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    cosine_angle = dot_product / (
        magnitude_ba * magnitude_bc
    )

    cosine_angle = max(
        -1,
        min(1, cosine_angle)
    )

    angle = math.degrees(
        math.acos(cosine_angle)
    )

    return angle



def calculate_shoulder_rotation(
        left_shoulder,
        right_shoulder):
    """
    Estimates shoulder rotation angle.

    Larger values indicate more torso rotation.
    """

    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]

    rotation = math.degrees(
        math.atan2(dy, dx)
    )

    return abs(rotation)



def calculate_hip_rotation(
        left_hip,
        right_hip):
    """
    Estimates hip rotation angle.
    """

    dx = right_hip[0] - left_hip[0]
    dy = right_hip[1] - left_hip[1]

    rotation = math.degrees(
        math.atan2(dy, dx)
    )

    return abs(rotation)



def calculate_head_stability(
        previous_head,
        current_head):
    """
    Measures head movement.

    Lower movement = better stability.

    Returns score from 0-100.
    """

    if previous_head is None:
        return 100


    dx = current_head[0] - previous_head[0]
    dy = current_head[1] - previous_head[1]


    movement = math.sqrt(
        dx ** 2 +
        dy ** 2
    )


    # Convert movement into score
    score = 100 - (movement * 5)


    score = max(
        0,
        min(100, score)
    )

    return score