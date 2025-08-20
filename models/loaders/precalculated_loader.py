import os
import pickle

from ultralytics.engine.results import Results  # pyright: ignore[reportMissingTypeStubs]


class PreCalculatedLoader:
    @staticmethod
    def load_results(
        input_folder: str,
        base_name_cls: str | None = "cls_results",
        base_name_det: str | None = "det_results",
        base_name_seg: str | None = "seg_results",
    ) -> dict[str, list[tuple[str, Results]] | None]:
        """
        Load the pre-calculated results from the specified folder.

        Args:
            input_folder (str): Path to the folder containing the pre-calculated results.
            base_name_cls (str | None): Base name for the classification results file.
            base_name_det (str | None): Base name for the detection results file.
            base_name_seg (str | None): Base name for the segmentation results file.

        Returns:
            dict[str, list[tuple[str, Results]] | None]: A dictionary containing the loaded results.
        """

        results: dict[str, list[tuple[str, Results]] | None] = {"cls": None, "det": None, "seg": None}

        if base_name_cls is not None:
            with open(file=os.path.join(input_folder, base_name_cls + ".pkl"), mode="rb") as f:
                results["cls"] = pickle.load(file=f)

        if base_name_det is not None:
            with open(file=os.path.join(input_folder, base_name_det + ".pkl"), mode="rb") as f:
                results["det"] = pickle.load(file=f)

        if base_name_seg is not None:
            with open(file=os.path.join(input_folder, base_name_seg + ".pkl"), mode="rb") as f:
                results["seg"] = pickle.load(file=f)

        return results
