import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.model.adversarial_model.base_encoder import BaseEncoder

class MaskDecoder(nn.Module):
    """
    Predicts:
        location_logits : where pixels should be selected
        color_delta     : how selected pixels should change
    """

    def __init__(self, in_channels: int):
        super().__init__()

        hidden = max(in_channels // 4, 64)

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.location_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.color_head = nn.Conv2d(hidden, 3, kernel_size=1)

        # Let the random high-resolution prior dominate initially.
        nn.init.normal_(self.location_head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.location_head.bias)

        # Start candidate colors close to the input image.
        nn.init.normal_(self.color_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.color_head.bias)

    def forward(self, features, output_size):
        features = self.shared(features)

        location_logits = self.location_head(features)
        color_delta = self.color_head(features)

        location_logits = F.interpolate(
            location_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        color_delta = F.interpolate(
            color_delta,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return location_logits, color_delta


class PatchMaskModel(nn.Module):
    def __init__(
        self,
        encoder: BaseEncoder,
        patch_size=(512, 512),
        epsilon: float = 0.10,
        mask_temperature: float = 0.10,
        selection_grid=64,
    ):
        super().__init__()

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")

        self.encoder = encoder
        self.epsilon = epsilon
        self.mask_temperature = mask_temperature
        self.selection_grid = selection_grid

        self.mask_decoder = MaskDecoder(
            in_channels=encoder.out_channels
        )

        height, width = patch_size

        # Randomly initialized spatial scores.
        # These provide scattered initial Top-K locations.
        self.location_prior = nn.Parameter(
            torch.randn(1, 1, height, width) * 0.02
        )

    def _create_mask(
        self,
        location_logits,
    ):
        batch_size, _, height, width = (
            location_logits.shape
        )

        grid = self.selection_grid

        cell_height = height // grid
        cell_width = width // grid

        num_cells = grid * grid
        pixels_per_cell = cell_height * cell_width

        total_selected = round(
            self.epsilon * height * width
        )

        base_selected = (
            total_selected // num_cells
        )

        remaining = (
            total_selected % num_cells
        )

        # Convert image into non-overlapping cells.
        cells = location_logits.reshape(
            batch_size,
            1,
            grid,
            cell_height,
            grid,
            cell_width,
        )

        cells = cells.permute(
            0,
            2,
            4,
            1,
            3,
            5,
        )

        cells = cells.reshape(
            batch_size,
            num_cells,
            pixels_per_cell,
        )

        max_selected = (
            base_selected
            + int(remaining > 0)
        )

        top_values, top_indices = torch.topk(
            cells,
            k=max_selected,
            dim=2,
        )

        hard_cells = torch.zeros_like(
            cells
        )

        if base_selected > 0:
            hard_cells.scatter_(
                dim=2,
                index=top_indices[
                    :,
                    :,
                    :base_selected
                ],
                value=1.0,
            )

        quota = torch.full(
            (
                batch_size,
                num_cells,
            ),
            base_selected,
            device=location_logits.device,
            dtype=torch.long,
        )

        # Some cells receive one additional selected pixel
        # so that the global count remains exact.
        if remaining > 0:
            extra_scores = top_values[
                :,
                :,
                base_selected
            ]

            extra_cells = torch.topk(
                extra_scores,
                k=remaining,
                dim=1,
            ).indices

            extra_pixels = top_indices[
                :,
                :,
                base_selected
            ].gather(
                dim=1,
                index=extra_cells,
            )

            batch_indices = torch.arange(
                batch_size,
                device=location_logits.device,
            ).unsqueeze(1)

            hard_cells[
                batch_indices,
                extra_cells,
                extra_pixels
            ] = 1.0

            quota.scatter_(
                dim=1,
                index=extra_cells,
                value=base_selected + 1,
            )

        threshold_index = (
            quota - 1
        ).clamp_min(0).unsqueeze(-1)

        threshold = top_values.gather(
            dim=2,
            index=threshold_index,
        ).detach()

        soft_cells = torch.sigmoid(
            (
                cells - threshold
            )
            / self.mask_temperature
        )

        mask_cells = (
            hard_cells
            + soft_cells
            - soft_cells.detach()
        )

        def restore_image(tensor):
            tensor = tensor.reshape(
                batch_size,
                grid,
                grid,
                1,
                cell_height,
                cell_width,
            )

            tensor = tensor.permute(
                0,
                3,
                1,
                4,
                2,
                5,
            )

            return tensor.reshape(
                batch_size,
                1,
                height,
                width,
            )

        mask = restore_image(
            mask_cells
        )

        hard_mask = restore_image(
            hard_cells
        )

        soft_mask = restore_image(
            soft_cells
        )

        return mask, hard_mask, soft_mask

    def forward(self, x, return_aux=False):
        batch_size, _, height, width = x.shape

        features = self.encoder(x)

        location_logits, color_delta = self.mask_decoder(
            features,
            output_size=(height, width),
        )

        location_prior = self.location_prior

        if location_prior.shape[-2:] != (height, width):
            location_prior = F.interpolate(
                location_prior,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        location_logits = location_logits + location_prior

        mask, hard_mask, soft_mask = self._create_mask(
            location_logits
        )

        # Image-conditioned candidate colors.
        # No fixed RGB perturbation bound is imposed.
        patch = torch.clamp(
            x + color_delta,
            min=0.0,
            max=1.0,
        )

        adversarial_image = x * (1.0 - mask) + patch * mask

        if not return_aux:
            return patch, mask

        return {
            "patch": patch,
            "mask": mask,
            "hard_mask": hard_mask,
            "soft_mask": soft_mask,
            "location_logits": location_logits,
            "location_scores": torch.sigmoid(location_logits),
            "color_delta": color_delta,
            "adversarial_image": adversarial_image,
        }