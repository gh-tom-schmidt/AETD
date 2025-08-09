# process a video to show the systems output

from units import Unit
from driver import VideoLoader, DebugView

video_input_path
video_output_path

detection_model_path
classification_model_path
segmentation_model_path


vl = VideoLoader(video_input_path, video_output_path)
unit = Unit()
debug_view = DebugView()

frames = vl.getBatch()
while frames is not None:
    for frame in frames:
        unit.process(frame)
        result = unit.getResult()
        processed_frames = debug_view.draw(frame, result)

    vl.saveBatch(processed_frames)
    frames = vl.getBatch()
