from modules.model.segmentation_model.base_segmentation_model import BaseSegmentationModel

from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
)


class SegFormerModel(BaseSegmentationModel):
    def __init__(
        self,
        model_name=(
            "nvidia/"
            "segformer-b0-finetuned-ade-512-512"
        ),
    ):
        super().__init__()

        self.processor = (
            AutoImageProcessor.from_pretrained(
                model_name
            )
        )

        self.model = (
            SegformerForSemanticSegmentation
            .from_pretrained(
                model_name
            )
        )

        self._num_classes = (
            self.model.config.num_labels
        )

        # ==========================================
        # Obtain preferred size automatically
        # ==========================================

        size = self.processor.size

        if isinstance(size, dict):

            if (
                "height" in size
                and
                "width" in size
            ):

                self._input_size = (
                    size["height"],
                    size["width"]
                )

            elif "shortest_edge" in size:

                edge = size[
                    "shortest_edge"
                ]

                self._input_size = (
                    edge,
                    edge
                )

            else:

                raise ValueError(
                    f"Unknown processor size "
                    f"format: {size}"
                )

        elif isinstance(
            size,
            (tuple, list)
        ):

            self._input_size = (
                size[-2],
                size[-1]
            )

        elif isinstance(size, int):

            self._input_size = (
                size,
                size
            )

        else:

            raise ValueError(
                f"Unsupported processor "
                f"size: {size}"
            )

        # ==========================================
        # Normalization
        # ==========================================

        self.register_buffer(
            "mean",
            torch.tensor(
                self.processor.image_mean
            ).view(
                1, 3, 1, 1
            )
        )

        self.register_buffer(
            "std",
            torch.tensor(
                self.processor.image_std
            ).view(
                1, 3, 1, 1
            )
        )


    @property
    def num_classes(self):
        return self._num_classes


    def forward(self, x):

        original_size = x.shape[-2:]

        x = F.interpolate(
            x,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )

        x = (
            x - self.mean
        ) / self.std

        outputs = self.model(
            pixel_values=x
        )

        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits