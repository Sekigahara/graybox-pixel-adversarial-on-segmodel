from modules.model.segmentation_model.base_segmentation_model import BaseSegmentationModel

import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

class DeepLabV3Model(BaseSegmentationModel):

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
                DeepLabV3_ResNet50_Weights.DEFAULT
            )

            self.model = deeplabv3_resnet50(
                weights=weights
            )

            self.categories = (
                weights.meta["categories"]
            )

        else:

            self.model = deeplabv3_resnet50(
                weights=None
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
        return self.model.classifier[0]
    
    def forward(self, x):

        """
        x:
            [B, 3, H, W]
            range [0, 1]

        returns:
            [B, 21, H, W]
        """

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

        output = self.model(x)

        logits = output["out"]

        logits = F.interpolate(
            logits,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits
