import os
import pickle

from ultralytics.engine.results import Results  # pyright: ignore[reportMissingTypeStubs]


class PreCalculatedLoader:
    @staticmethod
    def load_results(
        input_folder: str,
        base_name_cls: str | None = None,
        base_name_det: str | None = None,
        base_name_seg: str | None = None,
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
            path: str = os.path.join(input_folder, base_name_cls)
            try:
                with open(file=path, mode="rb") as f:
                    results["cls"] = pickle.load(file=f)
            except OSError:
                print(f"Error loading {path}. File not found or corrupted.")

            if not PreCalculatedLoader.has_type(results["cls"]):
                raise ValueError(f"Invalid format of {path}")

        if base_name_det is not None:
            path: str = os.path.join(input_folder, base_name_det)
            try:
                with open(file=path, mode="rb") as f:
                    results["det"] = pickle.load(file=f)
            except OSError:
                print(f"Error loading {path}. File not found or corrupted.")

            if not PreCalculatedLoader.has_type(results["det"]):
                raise ValueError(f"Invalid format of {path}")

        if base_name_seg is not None:
            path: str = os.path.join(input_folder, base_name_seg)
            try:
                with open(file=path, mode="rb") as f:
                    results["seg"] = pickle.load(file=f)
            except OSError:
                print(f"Error loading {path}. File not found or corrupted.")

            if not PreCalculatedLoader.has_type(results["seg"]):
                raise ValueError(f"Invalid format of {path}")

        return results

    @staticmethod
    def has_type(obj: object) -> bool:
        if isinstance(obj, list):
            for item in obj:  # type: ignore
                if not isinstance(item, tuple) or len(item) == 2:  # type: ignore
                    name: str = item[0]  # type: ignore
                    res: Results = item[1]  # type: ignore
                    if isinstance(name, str) and isinstance(res, Results):
                        return True
        return False
