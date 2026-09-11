import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.model.adversarial_model.base_encoder import BaseEncoder

class ViTEncoder(BaseEncoder):

    def __init__(
        self,
        model_name="vit_base_patch16_224",
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

        # ==================================================
        # Standard ViT:
        #
        # [B, N, D]
        # ==================================================

        if features.ndim == 3:

            # CLS token / register tokens / distillation token
            num_prefix = getattr(
                self.backbone,
                "num_prefix_tokens",
                1
            )

            tokens = features[:, num_prefix:, :]

            B, N, D = tokens.shape

            grid_size = self.backbone.patch_embed.grid_size

            h, w = grid_size

            if h * w != N:
                raise ValueError(
                    f"Token count {N} cannot be reshaped "
                    f"into grid {grid_size}"
                )

            features = tokens.transpose(
                1, 2
            ).reshape(
                B,
                D,
                h,
                w
            )

        # ==================================================
        # Some transformer implementations can return
        # [B, H, W, D]
        # ==================================================

        elif features.ndim == 4:

            if features.shape[-1] == self.out_channels:
                features = features.permute(
                    0, 3, 1, 2
                )

        else:
            raise ValueError(
                f"Unsupported ViT output: {features.shape}"
            )

        return features
