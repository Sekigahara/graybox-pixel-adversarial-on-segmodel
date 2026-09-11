import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.model.adversarial_model.base_encoder import BaseEncoder

class MaskDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
    ):
        super().__init__()

        hidden = max(
            in_channels // 4,
            64
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden,
                kernel_size=3,
                padding=1,
            ),

            nn.GELU(),

            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
            ),

            nn.GELU(),

            nn.Conv2d(
                hidden,
                1,
                kernel_size=1,
            ),
        )


    def forward(
        self,
        features,
        output_size,
    ):

        logits = self.decoder(
            features
        )

        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits
    
class PatchMaskModel(nn.Module):
    def __init__(
        self,
        encoder: BaseEncoder,
        patch_size=(512, 512),
        epsilon: float = 0.10,
    ):
        super().__init__()

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be between 0 and 1."
            )

        self.encoder = encoder
        self.epsilon = epsilon
        self.patch_size = patch_size

        # ----------------------------------------------
        # Mask decoder is still initialized internally
        # ----------------------------------------------

        self.mask_decoder = MaskDecoder(
            in_channels=encoder.out_channels
        )

        # ----------------------------------------------
        # Trainable patch
        #
        # Stored directly at segmentation resolution
        # ----------------------------------------------

        H, W = patch_size

        self.patch_logits = nn.Parameter(
            torch.zeros(
                1,
                3,
                H,
                W
            )
        )

        nn.init.normal_(
            self.patch_logits,
            mean=0.0,
            std=0.02,
        )

    def _create_mask(
        self,
        scores,
    ):

        """
        scores:
            [B, 1, H, W]

        Returns exact binary Top-K mask.
        """

        B, _, H, W = scores.shape

        total_pixels = H * W

        k = round(
            self.epsilon
            * total_pixels
        )

        if k == 0:
            return torch.zeros_like(scores)

        if k >= total_pixels:
            return torch.ones_like(scores)

        flat_scores = scores.flatten(
            start_dim=1
        )

        # Select exact K highest-scoring pixels
        topk_indices = torch.topk(
            flat_scores,
            k=k,
            dim=1,
        ).indices

        hard_mask = torch.zeros_like(
            flat_scores
        )

        hard_mask.scatter_(
            dim=1,
            index=topk_indices,
            value=1.0,
        )

        hard_mask = hard_mask.view(
            B,
            1,
            H,
            W
        )

        # Straight-through estimator
        #
        # Forward:
        #     binary mask
        #
        # Backward:
        #     gradients flow through scores

        mask = (
            hard_mask
            + scores
            - scores.detach()
        )

        return mask


    def forward(self, x):
        B, _, H, W = x.shape

        # ==============================================
        # Encoder
        #
        # encoder internally handles:
        # 512 -> 224
        # ==============================================

        features = self.encoder(x)

        # ==============================================
        # Mask
        # ==============================================

        mask_logits = self.mask_decoder(
            features,
            output_size=(H, W)
        )

        scores = torch.sigmoid(
            mask_logits
        )

        mask = self._create_mask(
            scores
        )

        # ==============================================
        # Patch
        # ==============================================

        patch = torch.sigmoid(
            self.patch_logits
        )

        # Normally unnecessary if dataset resolution
        # equals patch_size, but keeps it reusable.
        if patch.shape[-2:] != (H, W):

            patch = F.interpolate(
                patch,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )

        patch = patch.expand(
            B,
            -1,
            -1,
            -1
        )

        return patch, mask