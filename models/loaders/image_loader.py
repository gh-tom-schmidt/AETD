import os
from collections.abc import Iterator

import cv2
from cv2.typing import MatLike
from tqdm import tqdm


class ImageLoader:
    """
    Iterator for loading images from a directory.

    Args:
        input_folder (str): Path to the input folder containing images.
        show_progress (bool): Whether to show a progress bar while loading images.
    """

    def __init__(self, input_folder: str, show_progress: bool = True) -> None:
        """
        Iterator for loading images from a directory. The iterator returns the
        basename and the image.

        Args:
            input_folder (str): Path to the input folder containing images.
            show_progress (bool): Whether to show a progress bar while loading images.
        """

        # collect valid image file paths
        self.files: list[str] = [
            os.path.join(input_folder, f)
            for f in os.listdir(path=input_folder)
            if os.path.isfile(path=os.path.join(input_folder, f)) and f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.index = 0
        self.show_progress: bool = show_progress
        self._iterator: Iterator[str]

    def __iter__(self):
        self.index = 0
        if self.show_progress:
            self._iterator: Iterator[str] = iter(tqdm(iterable=self.files, desc="Loading images"))
        else:
            self._iterator: Iterator[str] = iter(self.files)
        return self

    def __next__(self) -> tuple[str, MatLike]:
        try:
            file_path: str = next(self._iterator)
        except StopIteration:
            raise StopIteration

        img: MatLike | None = cv2.imread(filename=file_path)
        basename: str = os.path.basename(file_path)

        if img is None:
            print(f"Warning: Could not read {file_path}. Skipping.")
            return self.__next__()

        return basename, img
