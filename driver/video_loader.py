import cv2
import concurrent.futures

class VideoLoader():
    def __init__(self, input_path, output_path):
        
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            print("Error opening video file")
            return
        
        self.input_path = input_path 
        self.output_path = output_path
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG', etc.

        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))


    def getbatch(self):
        while True:
            frames = []
            for _ in range(3):
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    break
                frames.append(frame)

            return frames
        
        return None

    def saveBatch(self, frames):
        for f in frames:
            self.out.write(f)
 
    def stop(self):
        self.cap.release()
        self.out.release()
        print("Saved processed video to: ", self.output_path)
