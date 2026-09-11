import json
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from inplace_abn import InPlaceABN

from modules.model.segmentation_model.base_segmentation_model import BaseSegmentationModel

# Change only these two imports if you store the official
# Mapillary InPlace-ABN source under another package path.
from third_party.mapillary_iabn.models.wider_resnet import net_wider_resnet38_a2
from third_party.mapillary_iabn.modules.deeplab import DeeplabV3

class WideResNet38DeepLab3VistasModel(
    BaseSegmentationModel
):
    def __init__(
        self,
        checkpoint_path,
        input_size=(512, 512),
        ignore_index=65,
        label_config_path=None,
    ):
        super().__init__()

        norm_act = partial(
            InPlaceABN,
            activation="leaky_relu",
            activation_param=0.01,
        )

        # WideResNet38 feature extractor
        self.body = net_wider_resnet38_a2(
            norm_act=norm_act,
            dilation=(1, 2, 4, 4),
        )

        # DeepLab3 semantic head
        self.head = DeeplabV3(
            in_channels=4096,
            out_channels=256,
            hidden_channels=256,
            norm_act=norm_act,
            pooling_size=(84, 84),
        )

        # Mapillary Vistas classifier
        self.classifier = nn.Conv2d(
            256,
            65,
            kernel_size=1,
        )

        self._input_size = tuple(input_size)
        self._num_classes = 65
        self._ignore_index = ignore_index

        self.register_buffer(
            "mean",
            torch.tensor(
                [
                    0.41738699,
                    0.45732192,
                    0.46886091,
                ],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [
                    0.25685097,
                    0.26509955,
                    0.29067996,
                ],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        state_dict = checkpoint["state_dict"]

        self.body.load_state_dict(
            state_dict["body"]
        )

        self.head.load_state_dict(
            state_dict["head"]
        )

        self.classifier.load_state_dict(
            state_dict["cls"]
        )

        self.class_names = {
            class_id: f"class_{class_id}"
            for class_id in range(
                self._num_classes
            )
        }

        if label_config_path is not None:
            with open(
                label_config_path,
                "r",
            ) as file:
                labels = json.load(file)["labels"]

            self.class_names = {
                class_id: labels[class_id]["readable"]
                for class_id in range(
                    self._num_classes
                )
            }

    def get_feature_module(self):
        """
        Capture the 256-channel DeepLab3 output before
        the final class projection.
        """

        return self.head

    def get_ignore_index(self):
        return self._ignore_index

    def get_class_names(self):
        return self.class_names

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

        x = (x - self.mean) / self.std

        features = self.body(x)

        semantic_features = self.head(
            features
        )

        logits = self.classifier(
            semantic_features
        )

        logits = F.interpolate(
            logits,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits