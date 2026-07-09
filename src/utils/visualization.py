import cv2



def draw_text(
        frame,
        text,
        position,
        color=(255, 255, 255)):
    """
    Draws readable text with shadow.
    """

    x, y = position


    # Shadow

    cv2.putText(
        frame,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2
    )


    # Text

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1
    )



def draw_dashboard(
        frame,
        swing_phase,
        swing_speed,
        left_elbow,
        right_elbow,
        head_stability,
        shoulder_rotation,
        hip_rotation
):
    """
    Compact CricketVision AI dashboard.
    """


    overlay = frame.copy()


    # Smaller dashboard

    cv2.rectangle(
        overlay,
        (10, 10),
        (300, 250),
        (0, 0, 0),
        -1
    )


    # Transparency

    alpha = 0.55

    frame = cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0
    )



    # Title

    cv2.putText(
        frame,
        "CRICKETVISION AI",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )



    metrics = [

        (
            f"Phase: {swing_phase}",
            (0,255,255)
        ),

        (
            f"Speed: {swing_speed:.1f}",
            (255,255,255)
        ),

        (
            f"L Elbow: {left_elbow:.0f}",
            (255,255,255)
        ),

        (
            f"R Elbow: {right_elbow:.0f}",
            (255,255,255)
        ),

        (
            f"Head: {head_stability:.0f}%",
            (0,255,0)
        ),

        (
            f"Shoulder: {shoulder_rotation:.0f}",
            (255,255,255)
        ),

        (
            f"Hip: {hip_rotation:.0f}",
            (255,255,255)
        )

    ]



    y = 65


    for text, color in metrics:

        draw_text(
            frame,
            text,
            (20, y),
            color
        )

        y += 25



    return frame