from ultralytics import YOLO



class BallDetector:
    """
    Cricket ball detection module.

    Handles:
    - Loading YOLO ball model
    - Detecting ball position
    - Returning coordinates
    """



    def __init__(self, model_path=None):

        self.model = None


        if model_path:

            print("Loading ball detection model...")

            self.model = YOLO(
                model_path
            )

            print("Ball detection model loaded!")



    # ---------------------------------
    # Detect cricket ball
    # ---------------------------------

    def detect_ball(self, frame):

        """
        Returns:

        (x, y)

        if ball detected.

        Otherwise:

        None
        """


        if self.model is None:

            return None



        results = self.model(
            frame,
            verbose=False
        )



        for result in results:


            boxes = result.boxes



            if boxes is None:

                continue



            if len(boxes) == 0:

                continue



            # Get highest confidence detection

            confidence_scores = boxes.conf



            best_index = confidence_scores.argmax()



            box = boxes.xyxy[
                best_index
            ]



            x1, y1, x2, y2 = map(
                int,
                box
            )



            center_x = int(
                (x1 + x2) / 2
            )


            center_y = int(
                (y1 + y2) / 2
            )



            return (
                center_x,
                center_y
            )



        return None