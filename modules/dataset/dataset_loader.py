from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

class VistasDatasetLoader(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "training",
        image_size=(512, 512),
    ):
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.image_size = image_size

        self.image_dir = (
            self.root
            / self.split
            / "images"
        )

        self.label_dir = (
            self.root
            / self.split
            / "labels"
        )

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: "
                f"{self.image_dir}"
            )

        if not self.label_dir.exists():
            raise FileNotFoundError(
                f"Label directory not found: "
                f"{self.label_dir}"
            )

        self.samples = self._find_samples()


    def _find_samples(self):
        valid_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
        }

        image_paths = sorted([
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower()
            in valid_extensions
        ])

        samples = []

        for image_path in image_paths:

            label_path = (
                self.label_dir
                / f"{image_path.stem}.png"
            )

            if not label_path.exists():
                raise FileNotFoundError(
                    f"Label not found for: "
                    f"{image_path.name}"
                )

            samples.append(
                (
                    image_path,
                    label_path
                )
            )

        if len(samples) == 0:
            raise RuntimeError(
                f"No samples found in: "
                f"{self.image_dir}"
            )

        return samples


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):

        image_path, label_path = (
            self.samples[index]
        )

        # ==========================================
        # Load
        # ==========================================

        image = Image.open(
            image_path
        ).convert("RGB")

        target = Image.open(
            label_path
        )

        # ==========================================
        # Resize
        # ==========================================

        if self.image_size is not None:

            image = TF.resize(
                image,
                self.image_size,
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
                antialias=True,
            )

            # Segmentation labels must use nearest
            # neighbor interpolation.
            target = TF.resize(
                target,
                self.image_size,
                interpolation=(
                    InterpolationMode.NEAREST
                ),
            )

        # ==========================================
        # Image
        #
        # PIL ->
        # float32 [3,H,W]
        # range [0,1]
        # ==========================================

        image = TF.to_tensor(
            image
        )

        # ==========================================
        # Segmentation label
        #
        # PIL ->
        # int64 [H,W]
        # ==========================================

        target = np.asarray(
            target,
            dtype=np.int64
        )

        target = torch.from_numpy(
            target.copy()
        ).long()

        return {
            "image": image,
            "target": target,
        }