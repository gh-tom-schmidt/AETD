# process a video to show the systems output

from modules import Unit
from driver import VideoLoader, DebugView

video_input_path = "data/tests/ETS2_60-FPS_2025-08-09_19-51-39_trimmed.mp4"
video_output_path = "data/tests/ETS2_60-FPS_2025-08-09_19-51-39_trimmed_processed.mp4"

detection_model_path = "models/yolo-detect-m_best_epochs-100_size-460-960_05-08-2025.pt"
classification_model_path = "models/yolo-cls-s_best_epochs-30_size-32-32_06-08-2025.pt"
segmentation_model_path = "models/yolo-seg-m_full-road_best_epochs-300_size-460-960_07-08-2025.pt"


vl = VideoLoader(video_input_path, video_output_path)
unit = Unit(detection_model_path, classification_model_path, segmentation_model_path)
debug_view = DebugView()

for frame in vl:
    unit.process(frame)
    result = unit.getResult()
    processed_frames = debug_view.draw(frame, result)
    vl.save(processed_frames)
