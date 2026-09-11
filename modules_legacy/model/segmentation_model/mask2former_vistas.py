import torch
import torch.nn.functional as F

from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
)

from modules.model.segmentation_model.base_segmentation_model import BaseSegmentationModel

class Mask2FormerVistasModel(
    BaseSegmentationModel
):
    def __init__(
        self,
        model_name=(
            "facebook/"
            "mask2former-swin-large-"
            "mapillary-vistas-semantic"
        ),
    ):
        super().__init__()

        self.processor = (
            AutoImageProcessor.from_pretrained(
                model_name
            )
        )

        self.model = (
            Mask2FormerForUniversalSegmentation
            .from_pretrained(
                model_name
            )
        )

        self._num_classes = (
            self.model.config.num_labels
        )

        self._ignore_index = (
            self.processor.ignore_index
        )

        # ==========================================
        # Obtain preferred input size
        # ==========================================

        size = self.processor.size

        if (
            hasattr(size, "height")
            and hasattr(size, "width")
        ):
            self._input_size = (
                int(size.height),
                int(size.width),
            )

        elif isinstance(size, dict):
            self._input_size = (
                int(size["height"]),
                int(size["width"]),
            )

        elif isinstance(
            size,
            (tuple, list)
        ):
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
                f"Unsupported processor size: "
                f"{size}"
            )

        # ==========================================
        # Normalization
        # ==========================================

        self.register_buffer(
            "mean",
            torch.tensor(
                self.processor.image_mean
            ).view(
                1,
                3,
                1,
                1,
            )
        )

        self.register_buffer(
            "std",
            torch.tensor(
                self.processor.image_std
            ).view(
                1,
                3,
                1,
                1,
            )
        )


    def get_feature_module(self):
        """
        Return a spatial feature module for the
        semantic activation-gradient loss.
        """

        return (
            self.model
            .model
            .pixel_level_module
            .decoder
            .mask_projection
        )


    def get_ignore_index(self):
        return self._ignore_index


    @property
    def num_classes(self):
        return self._num_classes


    def forward(self, x):

        original_size = x.shape[-2:]

        # ==========================================
        # Resize and normalize
        # ==========================================

        x = F.interpolate(
            x,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )

        x = (
            x - self.mean
        ) / self.std

        # ==========================================
        # Mask2Former forward
        # ==========================================

        outputs = self.model(
            pixel_values=x,
            return_dict=True,
        )

        # [B, Q, C + 1]
        class_query_logits = (
            outputs.class_queries_logits
        )

        # [B, Q, Hm, Wm]
        mask_query_logits = (
            outputs.masks_queries_logits
        )

        # ==========================================
        # Resize query masks to the model's
        # preferred working resolution
        # ==========================================

        mask_query_logits = F.interpolate(
            mask_query_logits,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )

        # Remove the final null/no-object class.
        #
        # [B, Q, C]
        class_probabilities = (
            class_query_logits
            .softmax(dim=-1)[..., :-1]
        )

        # [B, Q, H, W]
        mask_probabilities = (
            mask_query_logits.sigmoid()
        )

        # ==========================================
        # Query predictions -> dense semantic scores
        #
        # [B,Q,C] x [B,Q,H,W]
        # ->
        # [B,C,H,W]
        # ==========================================

        semantic_scores = torch.einsum(
            "bqc,bqhw->bchw",
            class_probabilities,
            mask_probabilities,
        )

        # ==========================================
        # Resize semantic scores to original input
        # ==========================================

        if (
            semantic_scores.shape[-2:]
            != original_size
        ):
            semantic_scores = F.interpolate(
                semantic_scores,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )

        # Mask2Former does not directly return dense
        # per-pixel class logits like SegFormer.
        #
        # Log semantic scores provide a compatible
        # dense representation for:
        #
        # F.cross_entropy()
        # segmentation_margin_loss()
        logits = torch.log(
            semantic_scores.clamp_min(
                1e-8
            )
        )

        return logits