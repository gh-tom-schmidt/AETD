import ast
import os
import pickle
import sys

from loaders import ImageLoader
from model import ClassificationModel, DetectionModel, SegmentationModel
from ultralytics.engine.results import Results  # pyright: ignore[reportMissingTypeStubs]


def save_results(
    cls_results: list[tuple[str, Results]],
    det_results: list[tuple[str, Results]],
    seg_results: list[tuple[str, Results]],
    output_folder: str,
) -> None:
    # check if output directory exists
    os.makedirs(name=output_folder, exist_ok=True)

    with open(file=os.path.join(output_folder, "cls_results.pkl"), mode="wb") as f:
        pickle.dump(obj=cls_results, file=f)

    with open(file=os.path.join(output_folder, "det_results.pkl"), mode="wb") as f:
        pickle.dump(obj=det_results, file=f)

    with open(file=os.path.join(output_folder, "seg_results.pkl"), mode="wb") as f:
        pickle.dump(obj=seg_results, file=f)


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(
            "Usage: python script.py\n"
            "\t<input_folder: str>\n"
            "\t<output_folder: str>\n"
            "\t<device: list[int] | str>\n"
            "\t<top_crop: int>\n"
            "\t<segmentation_model_path: str>\n"
            "\t<detection_model_path: str>\n"
            "\t<classification_model_path: str>"
        )
        sys.exit(1)

    # get input and output folders from command line
    input_folder: str = sys.argv[1]
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
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Device: {final_device}")
    print(f"Top crop: {top_crop}")
    print(f"Segmentation model path: {segmentation_model_path}")
    print(f"Detection model path: {detection_model_path}")
    print(f"Classification model path: {classification_model_path}")
    print("Loading models and images...")

    # get the images
    image_loader = ImageLoader(input_folder=input_folder)

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

    # generate the results for each image
    for basename, img in image_loader:
        # for now this only return one result because there is no batching
        img = img[top_crop:, :, :]

        cls_results.append((basename, classification_model.predict(img=img)))
        det_results.append((basename, detection_model.predict(img=img)))
        seg_results.append((basename, segmentation_model.predict(img=img)))

    save_results(cls_results=cls_results, det_results=det_results, seg_results=seg_results, output_folder=output_folder)
    print(f"Files saved to {output_folder}")
