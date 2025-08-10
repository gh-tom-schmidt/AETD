from modules import Unit
from driver import SampleLoader, DebugView

img_dir = "data/tests/Samples/"

detection_model_path = "models/yolo-detect-m_best_epochs-100_size-460-960_05-08-2025.pt"
classification_model_path = "models/yolo-cls-s_best_epochs-30_size-32-32_06-08-2025.pt"
segmentation_model_path = (
    "models/yolo-seg-m_full-road_best_epochs-300_size-460-960_07-08-2025.pt"
)


sl = SampleLoader(img_dir)
unit = Unit(detection_model_path, classification_model_path, segmentation_model_path)
debug_view = DebugView(manual_control=True)

for frame in sl:
    unit.process(frame)
    result = unit.getResult()
    processed_frame = debug_view.draw(frame, result)
    is_open = debug_view.window(processed_frame)
    if not is_open:
        break
