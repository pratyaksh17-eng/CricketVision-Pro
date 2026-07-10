import cv2
import os


class VideoProcessor:
    """
    Handles reading an input video and writing
    an annotated output video.
    """

    def __init__(self, input_path, output_path):

        self.input_path = input_path
        self.output_path = output_path

        self.cap = cv2.VideoCapture(input_path)

        if not self.cap.isOpened():
            raise FileNotFoundError(
                f"Cannot open video:\n{input_path}"
            )

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.fps <= 0:
            self.fps = 30

        self.frame_count = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        fourcc = cv2.VideoWriter_fourcc(*"avc1")

        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )

        self.current_frame = 0

    def read(self):

        success, frame = self.cap.read()

        if success:
            self.current_frame += 1

        return success, frame

    def write(self, frame):
        self.writer.write(frame)

    def progress(self):

        if self.frame_count == 0:
            return 0

        return (
            self.current_frame /
            self.frame_count
        ) * 100

    def release(self):

        self.cap.release()
        self.writer.release()

        cv2.destroyAllWindows()