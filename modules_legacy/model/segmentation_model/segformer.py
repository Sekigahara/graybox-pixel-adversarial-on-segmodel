import torch
import torch.nn.functional as F

from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
)

from modules.model.segmentation_model.base_segmentation_model import BaseSegmentationModel

class SegFormerModel(BaseSegmentationModel):
    def __init__(
        self,
        model_name=(
            "nvidia/"
            "segformer-b0-finetuned-ade-512-512"
        ),
    ):
        super().__init__()

        self.processor = (
            AutoImageProcessor.from_pretrained(
                model_name
            )
        )

        self.model = (
            SegformerForSemanticSegmentation
            .from_pretrained(
                model_name
            )
        )

        self._num_classes = (
            self.model.config.num_labels
        )

        # ==========================================
        # Obtain preferred size automatically
        # ==========================================

        size = self.processor.size

        if hasattr(size, "height") and hasattr(size, "width"):
            self._input_size = (
                int(size.height),
                int(size.width),
            )

        elif isinstance(size, dict):
            self._input_size = (
                int(size["height"]),
                int(size["width"]),
            )

        elif isinstance(size, (tuple, list)):
            self._input_size = (
                int(size[-2]),
                int(size[-1]),
            )

        elif isinstance(size, int):
            self._input_size = (
                size,
                size,
            )

        else:
            raise ValueError(
                f"Unsupported processor size: {size}"
            )

        # ==========================================
        # Normalization
        # ==========================================

        self.register_buffer(
            "mean",
            torch.tensor(
                self.processor.image_mean
            ).view(
                1, 3, 1, 1
            )
        )

        self.register_buffer(
            "std",
            torch.tensor(
                self.processor.image_std
            ).view(
                1, 3, 1, 1
            )
        )

    def get_feature_module(self):
        return (
            self.model
            .decode_head
            .activation
        )
        
    @property
    def num_classes(self):
        return self._num_classes


    def forward(self, x):

        original_size = x.shape[-2:]

        x = F.interpolate(
            x,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )

        x = (
            x - self.mean
        ) / self.std

        outputs = self.model(
            pixel_values=x
        )

        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits
