import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class BaseEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        freeze: bool = False,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
        )

        self.out_channels = self.backbone.num_features

        # ==========================================
        # Read configuration from pretrained model
        # ==========================================

        cfg = self.backbone.pretrained_cfg

        input_size = cfg.get(
            "input_size",
            (3, 224, 224)
        )

        # timm returns:
        #
        # (C, H, W)
        #
        # We only expose:
        #
        # (H, W)

        self._input_size = (
            input_size[-2],
            input_size[-1]
        )

        mean = cfg.get(
            "mean",
            (0.485, 0.456, 0.406)
        )

        std = cfg.get(
            "std",
            (0.229, 0.224, 0.225)
        )

        self.register_buffer(
            "mean",
            torch.tensor(mean).view(
                1, 3, 1, 1
            )
        )

        self.register_buffer(
            "std",
            torch.tensor(std).view(
                1, 3, 1, 1
            )
        )

        if freeze:

            for param in self.backbone.parameters():
                param.requires_grad = False

    # ==============================================
    # Public input-size interface
    # ==============================================

    @property
    def input_size(self):
        return self._input_size


    def get_input_size(self):
        return self._input_size


    def preprocess(self, x):

        x = F.interpolate(
            x,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )

        x = (
            x - self.mean
        ) / self.std

        return x


    def forward(self, x):
        raise NotImplementedError