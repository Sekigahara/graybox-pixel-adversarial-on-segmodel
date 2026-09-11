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

    def get_feature_layers(self):
        backbone = self.backbone

        # ==============================================
        # ViT / MobileNet / EfficientNet-like blocks
        # ==============================================

        if hasattr(backbone, "blocks"):

            blocks = list(
                backbone.blocks.children()
            )

            if len(blocks) > 0:
                return [
                    [block]
                    for block in blocks
                ]

        # ==============================================
        # ConvNeXt
        # ==============================================

        if hasattr(backbone, "stages"):

            stages = list(
                backbone.stages.children()
            )

            if len(stages) > 0:
                return [
                    [stage]
                    for stage in stages
                ]

        # ==============================================
        # ResNet
        # ==============================================

        resnet_stages = []

        for name in [
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            if hasattr(backbone, name):

                resnet_stages.append([
                    getattr(
                        backbone,
                        name
                    )
                ])

        if len(resnet_stages) > 0:
            return resnet_stages

        # ==============================================
        # VGG
        # ==============================================

        if hasattr(backbone, "features"):

            modules = list(
                backbone.features.children()
            )

            if any(
                isinstance(m, nn.MaxPool2d)
                for m in modules
            ):
                stages = []
                current = []

                for module in modules:
                    current.append(
                        module
                    )

                    if isinstance(
                        module,
                        nn.MaxPool2d
                    ):
                        stages.append(
                            current
                        )

                        current = []

                if current:
                    stages.append(
                        current
                    )

                return stages

        raise ValueError(
            "Unsupported encoder structure: "
            f"{backbone.__class__.__name__}"
        )
    
    def set_trainable_last_layers(
        self,
        num_layers,
    ):
        if num_layers < 0:

            raise ValueError(
                "num_layers must be >= 0."
            )

        feature_layers = (
            self.get_feature_layers()
        )

        total_layers = len(
            feature_layers
        )

        if num_layers > total_layers:

            raise ValueError(
                f"Requested {num_layers} trainable "
                f"layers but encoder only exposes "
                f"{total_layers} feature stages."
            )

        # ==============================================
        # Freeze everything
        # ==============================================

        for parameter in (
            self.parameters()
        ):

            parameter.requires_grad = False

        # ==============================================
        # Select deepest stages
        # ==============================================

        trainable_layers = (
            feature_layers[
                -num_layers:
            ]
            if num_layers > 0
            else []
        )

        # ==============================================
        # Unfreeze selected modules
        # ==============================================

        for stage in trainable_layers:

            for module in stage:

                for parameter in (
                    module.parameters()
                ):

                    parameter.requires_grad = True

        self._feature_layers = (
            feature_layers
        )

        self._trainable_feature_layers = (
            trainable_layers
        )

        return {
            "total_layers":
                total_layers,

            "trainable_layers":
                num_layers,

            "frozen_layers":
                total_layers - num_layers,
        }
    
    def set_finetune_mode(self):
        # Entire encoder frozen/eval first
        self.eval()

        if not hasattr(
            self,
            "_trainable_feature_layers"
        ):
            return

        # Only selected stages go back to train mode
        for stage in (
            self._trainable_feature_layers
        ):

            for module in stage:

                module.train()
    
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