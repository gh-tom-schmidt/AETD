import cv2
from tqdm import tqdm

class VideoLoader:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

        self.pbar = tqdm(total=self.total_frames, desc="Processing frames", unit="frame")

        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.writer.release()
            self.pbar.close()
            raise StopIteration
        self.pbar.update(1)
        return frame

    def save(self, frame):
        self.writer.write(frame)


