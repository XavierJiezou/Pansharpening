import codecs
import os
import os.path
import shutil
import string
import sys
import warnings
from glob import glob
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import URLError

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset


class PanBench(VisionDataset):
    """`PanBench <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/raw/train-images-idx3-ubyte``
            and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    # mirrors = [
    #     "http://yann.lecun.com/exdb/mnist/",
    #     "https://ossci-datasets.s3.amazonaws.com/mnist/",
    # ]

    # resources = [
    #     ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    #     ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    #     ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    #     ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    # ]

    # training_file = "training.pt"
    # test_file = "test.pt"
    # classes = [
    #     "0 - zero",
    #     "1 - one",
    #     "2 - two",
    #     "3 - three",
    #     "4 - four",
    #     "5 - five",
    #     "6 - six",
    #     "7 - seven",
    #     "8 - eight",
    #     "9 - nine",
    # ]

    # @property
    # def train_labels(self):
    #     warnings.warn("train_labels has been renamed targets")
    #     return self.targets

    # @property
    # def test_labels(self):
    #     warnings.warn("test_labels has been renamed targets")
    #     return self.targets

    # @property
    # def train_data(self):
    #     warnings.warn("train_data has been renamed data")
    #     return self.data

    # @property
    # def test_data(self):
    #     warnings.warn("test_data has been renamed data")
    #     return self.data

    def __init__(
        self,
        root: str,
        band: str = "RGBN",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.band = band  # which band to use: "R", "G", "B", "N", "RGB", "RGBN"
        self.folder = os.path.join(self.root, self.__class__.__name__)
        self.rgb_paths = sorted(glob(os.path.join(self.folder, "*/*RGB*/*.tif")))
        self.nir_paths = sorted(glob(os.path.join(self.folder, "*/*NIR*/*.tif")))
        self.pan_paths = sorted(glob(os.path.join(self.folder, "*/*PAN*/*.tif")))

        # if self._check_legacy_exist():
        #     self.data, self.targets = self._load_legacy_data()
        #     return

        # if download:
        #     self.download()

        # if not self._check_exists():
        #     raise RuntimeError("Dataset not found. You can use download=True to download it")

        # self.data, self.targets = self._load_data()

    # def _check_legacy_exist(self):
    #     processed_folder_exists = os.path.exists(self.processed_folder)
    #     if not processed_folder_exists:
    #         return False

    #     return all(
    #         check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
    #     )

    # def _load_legacy_data(self):
    #     # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
    #     # directly.
    #     data_file = self.training_file if self.train else self.test_file
    #     return torch.load(os.path.join(self.processed_folder, data_file))

    # def _load_data(self):
    #     image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
    #     data = read_image_file(os.path.join(self.raw_folder, image_file))

    #     label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
    #     targets = read_label_file(os.path.join(self.raw_folder, label_file))

    #     return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        rgb = Image.open(self.rgb_paths[index])
        lrrgb = rgb.resize((int(rgb.size[0] / 4), int(rgb.size[1] / 4)), Image.Resampling.BICUBIC)
        nir = Image.open(self.nir_paths[index])
        lrnir = nir.resize((int(rgb.size[0] / 4), int(rgb.size[1] / 4)), Image.Resampling.BICUBIC)
        pan = Image.open(self.pan_paths[index])
        lrpan = pan.resize((int(pan.size[0] / 4), int(pan.size[1] / 4)), Image.Resampling.BICUBIC)
        red, green, blue = rgb.split()
        lrred = red.resize((int(rgb.size[0] / 4), int(rgb.size[1] / 4)), Image.Resampling.BICUBIC)
        lrgreen = green.resize(
            (int(rgb.size[0] / 4), int(rgb.size[1] / 4)), Image.Resampling.BICUBIC
        )
        lrblue = blue.resize(
            (int(rgb.size[0] / 4), int(rgb.size[1] / 4)), Image.Resampling.BICUBIC
        )

        if self.transform is not None:
            rgb = self.transform(rgb)
            lrrgb = self.transform(lrrgb)
            nir = self.transform(nir)
            lrnir = self.transform(lrnir)
            red = self.transform(red)
            lrred = self.transform(lrred)
            green = self.transform(green)
            lrgreen = self.transform(lrgreen)
            blue = self.transform(blue)
            lrblue = self.transform(lrblue)

        if self.target_transform is not None:
            pan = self.target_transform(pan)
            lrpan = self.target_transform(lrpan)

        if self.band == "R":
            ms = red
            lrms = lrred
        elif self.band == "G":
            ms = green
            lrms = lrgreen
        elif self.band == "B":
            ms = blue
            lrms = lrblue
        elif self.band == "RGB":
            ms = rgb
            lrms = lrrgb
        elif self.band == "RGBN":
            ms = torch.cat((rgb, nir))
            lrms = torch.cat((lrrgb, lrnir))
        else:
            raise "Unrecognized band type."

        return {
            "ms": ms,
            "lrms": lrms,
            "pan": pan,
            "lrpan": lrpan,
            "sate": self.rgb_paths[index].split(os.sep)[-3],
            "scene": self.rgb_paths[index].split(os.sep)[-1].split("_")[1][:-4],
        }

    def __len__(self) -> int:
        return len(self.rgb_paths)

    # @property
    # def raw_folder(self) -> str:
    #     return os.path.join(self.root, self.__class__.__name__, "raw")

    # @property
    # def processed_folder(self) -> str:
    #     return os.path.join(self.root, self.__class__.__name__, "processed")

    # @property
    # def class_to_idx(self) -> Dict[str, int]:
    #     return {_class: i for i, _class in enumerate(self.classes)}

    # def _check_exists(self) -> bool:
    #     return all(
    #         check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
    #         for url, _ in self.resources
    #     )

    # def download(self) -> None:
    #     """Download the MNIST data if it doesn't exist already."""

    #     if self._check_exists():
    #         return

    #     os.makedirs(self.raw_folder, exist_ok=True)

    #     # download files
    #     for filename, md5 in self.resources:
    #         for mirror in self.mirrors:
    #             url = f"{mirror}{filename}"
    #             try:
    #                 print(f"Downloading {url}")
    #                 download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)
    #             except URLError as error:
    #                 print(f"Failed to download (trying next):\n{error}")
    #                 continue
    #             finally:
    #                 print()
    #             break
    #         else:
    #             raise RuntimeError(f"Error downloading {filename}")

    # def extra_repr(self) -> str:
    #     split = "Train" if self.train is True else "Test"
    #     return f"Split: {split}"


if __name__ == "__main__":
    dataset = PanBench(
        root="data",
        band="RGBN",
        transform=transforms.ToTensor(),
        target_transform=transforms.ToTensor(),
    )
    for sample in dataset:
        assert sample["ms"].shape == (4, 256, 256)
        assert sample["lrms"].shape == (4, 64, 64)
        assert sample["pan"].shape == (1, 1024, 1024)
        assert sample["lrpan"].shape == (1, 256, 256)
        assert sample["sate"] in [
            "GF1",
            "GF2",
            "GF6",
            "IN",
            "LC7",
            "LC8",
            "QB",
            "WV2",
            "WV3",
            "WV4",
        ]
        assert sample["scene"].split(",")[0] in [
            "urban",
            "water",
            "crops",
            "ice",
            "vegetation",
            "barren",
        ]
        break
