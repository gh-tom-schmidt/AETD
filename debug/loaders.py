import cv2
from tqdm import tqdm
import random
import os


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

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )

        self.pbar = tqdm(
            total=self.total_frames, desc="Processing frames", unit="frame"
        )

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


class SampleLoader:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        if not self.image_files:
            raise ValueError("No images found in the directory.")

    def __iter__(self):
        self._shuffled_files = random.sample(self.image_files, len(self.image_files))
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self._shuffled_files):
            raise StopIteration
        img_path = os.path.join(self.image_dir, self._shuffled_files[self._index])
        image = cv2.imread(img_path)
        if image is None:
            self._index += 1
            return self.__next__()
        self._index += 1
        return image
