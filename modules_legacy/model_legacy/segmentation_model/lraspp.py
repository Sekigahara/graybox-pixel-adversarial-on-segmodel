from modules.model.segmentation_model.base_segmentation_model import BaseSegmentationModel

import torch
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights,
)


class LRASPPModel(BaseSegmentationModel):

    def __init__(
        self,
        pretrained=True,
        resize_input=True,
        input_size=(520, 520),
    ):
        super().__init__()
        
        self._input_size = input_size

        if pretrained:

            weights = (
                LRASPP_MobileNet_V3_Large_Weights.DEFAULT
            )

            self.model = (
                lraspp_mobilenet_v3_large(
                    weights=weights
                )
            )

            self.categories = (
                weights.meta["categories"]
            )

        else:

            self.model = (
                lraspp_mobilenet_v3_large(
                    weights=None
                )
            )

            self.categories = None

        self.resize_input = resize_input

        self.register_buffer(
            "mean",
            torch.tensor([
                0.485,
                0.456,
                0.406
            ]).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor([
                0.229,
                0.224,
                0.225
            ]).view(1, 3, 1, 1)
        )


    @property
    def num_classes(self):
        return 21

    def get_feature_module(self):
        return (
            self.model
            .classifier
            .cbr
        )

    def forward(self, x):
        original_size = x.shape[-2:]

        if self.resize_input:

            x = F.interpolate(
                x,
                size=(520, 520),
                mode="bilinear",
                align_corners=False,
            )

        x = (
            x - self.mean
        ) / self.std

        logits = self.model(x)["out"]

        logits = F.interpolate(
            logits,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits