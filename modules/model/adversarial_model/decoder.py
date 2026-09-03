import torch
import torch.nn as nn
import torch.nn.functional as F

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
        epsilon: float = 0.10,
    ):
        """
        epsilon:
            fixed fraction of pixels that can be modified.

        Example:

            epsilon = 0.10

        means exactly 10% of the spatial pixels
        are selected by the mask.
        """

        super().__init__()

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be between 0 and 1."
            )

        self.encoder = encoder

        # Fixed variable.
        # This is NOT trainable.
        self.epsilon = epsilon

        # ----------------------------------------------
        # Mask prediction
        # ----------------------------------------------

        self.mask_decoder = MaskDecoder(
            in_channels=encoder.out_channels
        )

        # ----------------------------------------------
        # Global trainable patch
        #
        # Patch stored at encoder's native resolution.
        # ----------------------------------------------

        H, W = encoder.input_size

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

        # ----------------------------------
        # Encoder handles its own resize
        # e.g. 512 -> 224
        # ----------------------------------

        features = self.encoder(x)

        # ----------------------------------
        # Decode features back to the
        # actual image / segmentation size
        # ----------------------------------

        mask_logits = self.mask_decoder(
            features,
            output_size=(H, W)
        )

        scores = torch.sigmoid(
            mask_logits
        )

        # epsilon calculated at H × W
        mask = self._create_mask(
            scores
        )

        # ----------------------------------
        # Patch
        # ----------------------------------

        patch = torch.sigmoid(
            self.patch_logits
        )

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