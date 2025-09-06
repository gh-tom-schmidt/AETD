import ast
import os
import pickle
import sys

import cv2
from model import ClassificationModel, DetectionModel, SegmentationModel
from tqdm import tqdm
from ultralytics.engine.results import Results  # pyright: ignore[reportMissingTypeStubs]


def save_results(
    basename: str,
    cls_results: list[tuple[str, Results]],
    det_results: list[tuple[str, Results]],
    seg_results: list[tuple[str, Results]],
    output_folder: str,
) -> None:
    # check if output directory exists
    os.makedirs(name=output_folder, exist_ok=True)

    with open(file=os.path.join(output_folder, f"{basename}_cls_results.pkl"), mode="wb") as f:
        pickle.dump(obj=cls_results, file=f)

    with open(file=os.path.join(output_folder, f"{basename}_det_results.pkl"), mode="wb") as f:
        pickle.dump(obj=det_results, file=f)

    with open(file=os.path.join(output_folder, f"{basename}_seg_results.pkl"), mode="wb") as f:
        pickle.dump(obj=seg_results, file=f)


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(
            "Usage: python script.py\n"
            "\t<video_source: str>\n"
            "\t<output_folder: str>\n"
            "\t<device: list[int] | str>\n"
            "\t<top_crop: int>\n"
            "\t<segmentation_model_path: str>\n"
            "\t<detection_model_path: str>\n"
            "\t<classification_model_path: str>"
        )
        sys.exit(1)

    # get input and output folders from command line
    video_src: str = sys.argv[1]
    output_folder: str = sys.argv[2]
    device: str = sys.argv[3]
    top_crop: int = int(sys.argv[4])
    segmentation_model_path: str = sys.argv[5]
    detection_model_path: str = sys.argv[6]
    classification_model_path: str = sys.argv[7]

    # check if the provided device is a string type
    # and if not than make a list
    if device not in ["cpu", "cuda"]:
        final_device: list[int] | str = ast.literal_eval(node_or_string=device)
    else:
        final_device: list[int] | str = device

    # the recognized command line inputs
    print(f"Video Source: {video_src}")
    print(f"Output folder: {output_folder}")
    print(f"Device: {final_device}")
    print(f"Top crop: {top_crop}")
    print(f"Segmentation model path: {segmentation_model_path}")
    print(f"Detection model path: {detection_model_path}")
    print(f"Classification model path: {classification_model_path}")
    print("Loading models and images...")

    # get the images
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # load the models
    classification_model = ClassificationModel(
        pretrained_model_path=classification_model_path,
        device=final_device,
    )
    detection_model = DetectionModel(
        pretrained_model_path=detection_model_path,
        device=final_device,
    )
    segmentation_model = SegmentationModel(
        pretrained_model_path=segmentation_model_path,
        device=final_device,
    )

    cls_results: list[tuple[str, Results]] = []
    det_results: list[tuple[str, Results]] = []
    seg_results: list[tuple[str, Results]] = []

    print("Everything loaded. Process images....")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # tqdm loop
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        fram_num = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

        frame = frame[top_crop:, :, :]

        cls_results.append((fram_num, classification_model.predict(img=frame)))
        det_results.append((fram_num, detection_model.predict(img=frame)))
        seg_results.append((fram_num, segmentation_model.predict(img=frame)))

    basename: str = os.path.splitext(os.path.basename(video_src))[0]
    save_results(
        basename=basename,
        cls_results=cls_results,
        det_results=det_results,
        seg_results=seg_results,
        output_folder=output_folder,
    )
    print(f"File saved to {output_folder}")
