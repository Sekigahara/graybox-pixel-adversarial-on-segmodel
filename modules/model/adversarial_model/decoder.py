import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.model.adversarial_model.base_encoder import BaseEncoder

class MaskDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
    ):
        super().__init__()

        groups = 8 if hidden_channels % 8 == 0 else 1

        self.semantic_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                groups,
                hidden_channels,
            ),
            nn.GELU(),
        )

        self.local_branch = nn.Sequential(
            nn.Conv2d(
                3,
                8,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.GroupNorm(4, 8),
            nn.GELU(),

            nn.Conv2d(
                8,
                8,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.GroupNorm(4, 8),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                hidden_channels + 8,
                hidden_channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.GroupNorm(
                groups,
                hidden_channels,
            ),
            nn.GELU(),
        )

        self.retention_head = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
        )

        self.delta_head = nn.Conv2d(
            hidden_channels,
            3,
            kernel_size=1,
        )

        nn.init.normal_(
            self.retention_head.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(
            self.retention_head.bias
        )

        nn.init.normal_(
            self.delta_head.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(
            self.delta_head.bias
        )


    def forward(
        self,
        features,
        image,
    ):
        output_size = image.shape[-2:]

        semantic_features = self.semantic_branch(
            features
        )

        semantic_features = F.interpolate(
            semantic_features,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        local_features = self.local_branch(
            image
        )

        fused_features = torch.cat(
            [
                semantic_features,
                local_features,
            ],
            dim=1,
        )

        fused_features = self.fusion(
            fused_features
        )

        retention_logits = self.retention_head(
            fused_features
        )

        raw_delta = self.delta_head(
            fused_features
        )

        return retention_logits, raw_delta

class PatchMaskModel(nn.Module):
    def __init__(
        self,
        encoder: BaseEncoder,
        patch_size=(512, 512),
        epsilon: float = 0.01,
        mask_temperature: float = 0.10,
        decoder_hidden_channels: int = 32,
    ):
        super().__init__()

        self.encoder = encoder
        self.patch_size = patch_size
        self.epsilon = epsilon
        self.mask_temperature = mask_temperature

        self.mask_decoder = MaskDecoder(
            in_channels=encoder.out_channels,
            hidden_channels=decoder_hidden_channels,
        )

    def _standardize_retention_logits(
        self,
        retention_logits,
    ):
        mean = retention_logits.mean(
            dim=(2, 3),
            keepdim=True,
        )

        std = retention_logits.std(
            dim=(2, 3),
            keepdim=True,
            unbiased=False,
        )

        return (
            retention_logits - mean
        ) / (
            std + 1e-6
        )


    def _create_mask(
        self,
        retention_logits,
        keep_ratio,
        use_hard_mask,
    ):
        batch_size, _, height, width = retention_logits.shape

        total_pixels = height * width
        k = round(
            keep_ratio * total_pixels
        )

        if k <= 0:
            zero_mask = torch.zeros_like(
                retention_logits
            )
            return zero_mask, zero_mask, zero_mask

        if k >= total_pixels:
            one_mask = torch.ones_like(
                retention_logits
            )
            return one_mask, one_mask, one_mask

        flat_logits = retention_logits.flatten(
            start_dim=1
        )

        top_values, top_indices = torch.topk(
            flat_logits,
            k=k,
            dim=1,
        )

        hard_mask = torch.zeros_like(
            flat_logits
        )

        hard_mask.scatter_(
            dim=1,
            index=top_indices,
            value=1.0,
        )

        threshold = top_values[:, -1:].detach()

        soft_mask = torch.sigmoid(
            (
                flat_logits - threshold
            ) / self.mask_temperature
        )

        if use_hard_mask:
            mask = (
                hard_mask
                + soft_mask
                - soft_mask.detach()
            )
        else:
            mask = soft_mask

        mask = mask.view(
            batch_size,
            1,
            height,
            width,
        )

        hard_mask = hard_mask.view(
            batch_size,
            1,
            height,
            width,
        )

        soft_mask = soft_mask.view(
            batch_size,
            1,
            height,
            width,
        )

        return mask, hard_mask, soft_mask

    def forward(
        self,
        x,
        keep_ratio=None,
        use_hard_mask=True,
        return_aux=False,
    ):
        if keep_ratio is None:
            keep_ratio = self.epsilon

        features = self.encoder(x)

        retention_logits, raw_delta = self.mask_decoder(
            features,
            x,
        )

        retention_logits = self._standardize_retention_logits(
            retention_logits
        )

        base_logits = torch.logit(
            x.clamp(
                min=1e-4,
                max=1.0 - 1e-4,
            )
        )

        patch = torch.sigmoid(
            base_logits + raw_delta
        )

        dense_delta = patch - x

        mask, hard_mask, soft_mask = self._create_mask(
            retention_logits,
            keep_ratio,
            use_hard_mask,
        )

        adversarial_image = (
            x + mask * dense_delta
        )

        if not return_aux:
            return patch, mask

        return {
            "patch": patch,
            "mask": mask,
            "hard_mask": hard_mask,
            "soft_mask": soft_mask,
            "retention_logits": retention_logits,
            "retention_scores": torch.sigmoid(
                retention_logits
            ),
            "raw_delta": raw_delta,
            "dense_delta": dense_delta,
            "adversarial_image": adversarial_image,
            "keep_ratio": keep_ratio,
        }
