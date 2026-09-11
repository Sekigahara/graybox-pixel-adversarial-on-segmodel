import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.model.adversarial_model.base_encoder import BaseEncoder

class ConvEncoder(BaseEncoder):
    def __init__(
        self,
        model_name="resnet50",
        pretrained=True,
        freeze=False,
    ):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            freeze=freeze,
        )
        
    def forward(self, x):
        x = self.preprocess(x)

        features = self.backbone.forward_features(x)
        
        # Check dim
        if features.ndim != 4:
            raise ValueError(
                f"Expected CNN feature map, "
                f"but received {features.shape}"
            )

        # Check channel order, Some architectures may return NHWC
        if (
            features.shape[1] != self.out_channels
            and features.shape[-1] == self.out_channels
        ):
            features = features.permute(
                0, 3, 1, 2
            )

        return features
