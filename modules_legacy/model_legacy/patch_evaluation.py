import json
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from modules.model.adversarial_model.conv_encoder import ConvEncoder
from modules.model.adversarial_model.vit_encoder import ViTEncoder
from modules.model.adversarial_model.decoder import PatchMaskModel


def load_patch_model(
    checkpoint_path,
    config,
    segmentation_model,
    device,
):
    """
    Rebuild PatchMaskModel from config and load its weights.

    Supports:
        - standalone export with "model_state_dict"
        - trainer checkpoint with "patch_model"
        - raw state_dict
    """

    encoder_type = config["encoder_model"]["type"]
    encoder_name = config["encoder_model"]["name"]

    if encoder_type == "conv":
        encoder = ConvEncoder(
            model_name=encoder_name,
            pretrained=False,
        )

    elif encoder_type == "vit":
        encoder = ViTEncoder(
            model_name=encoder_name,
            pretrained=False,
        )

    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}"
        )

    patch_model = PatchMaskModel(
        encoder=encoder,
        patch_size=segmentation_model.get_input_size(),
        epsilon=config["patch_model"]["eps"],
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "patch_model" in checkpoint:
        state_dict = checkpoint["patch_model"]
    else:
        state_dict = checkpoint

    patch_model.load_state_dict(
        state_dict,
        strict=True,
    )

    patch_model = patch_model.to(device)
    patch_model.eval()

    return patch_model


def resize_target_to_logits(
    target,
    logits,
):
    if target.shape[-2:] == logits.shape[-2:]:
        return target

    return (
        F.interpolate(
            target.unsqueeze(1).float(),
            size=logits.shape[-2:],
            mode="nearest",
        )
        .squeeze(1)
        .long()
    )


def update_confusion_matrix(
    confusion,
    prediction,
    target,
    num_classes,
):
    valid = (
        (target >= 0)
        &
        (target < num_classes)
    )

    if not valid.any():
        return confusion

    indices = (
        target[valid] * num_classes
        + prediction[valid]
    )

    count = torch.bincount(
        indices,
        minlength=num_classes ** 2,
    )

    confusion += (
        count
        .reshape(
            num_classes,
            num_classes,
        )
        .cpu()
    )

    return confusion


def compute_metrics(
    confusion,
):
    confusion = confusion.float()

    intersection = torch.diag(
        confusion
    )

    total = confusion.sum()

    pixel_accuracy = (
        intersection.sum()
        /
        total.clamp_min(1)
    )

    union = (
        confusion.sum(dim=1)
        +
        confusion.sum(dim=0)
        -
        intersection
    )

    valid = union > 0

    class_iou = torch.zeros_like(
        union
    )

    class_iou[valid] = (
        intersection[valid]
        /
        union[valid]
    )

    miou = (
        class_iou[valid].mean()
        if valid.any()
        else torch.tensor(0.0)
    )

    return {
        "pixel_accuracy":
            float(pixel_accuracy.item()),

        "miou":
            float(miou.item()),
    }


def build_palette(
    num_classes,
):
    palette = np.zeros(
        (num_classes, 3),
        dtype=np.uint8,
    )

    hue = 0.0
    ratio = 0.618033988749895

    for idx in range(num_classes):
        hue = (hue + ratio) % 1.0

        rgb = colorsys.hsv_to_rgb(
            hue,
            0.70,
            0.95,
        )

        palette[idx] = (
            np.asarray(rgb) * 255
        ).astype(np.uint8)

    if num_classes > 0:
        palette[0] = [0, 0, 0]

    return palette


def tensor_to_pil(
    image,
):
    image = (
        image.detach()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )

    return Image.fromarray(
        (image * 255)
        .round()
        .astype(np.uint8)
    )


def segmentation_to_pil(
    prediction,
    image_size,
    palette,
):
    prediction = (
        F.interpolate(
            prediction[None, None].float(),
            size=image_size,
            mode="nearest",
        )
        .squeeze()
        .long()
        .cpu()
        .numpy()
    )

    prediction = np.clip(
        prediction,
        0,
        len(palette) - 1,
    )

    return Image.fromarray(
        palette[prediction]
    )


def add_title(
    image,
    title,
):
    header = 30

    panel = Image.new(
        "RGB",
        (
            image.width,
            image.height + header,
        ),
        "white",
    )

    panel.paste(
        image,
        (0, header),
    )

    draw = ImageDraw.Draw(
        panel
    )

    draw.text(
        (8, 8),
        title,
        fill="black",
    )

    return panel


def save_comparison(
    clean_image,
    clean_prediction,
    patched_image,
    patched_prediction,
    palette,
    save_path,
):
    image_size = clean_image.shape[-2:]

    panels = [
        add_title(
            tensor_to_pil(clean_image),
            "Clean Image",
        ),

        add_title(
            segmentation_to_pil(
                clean_prediction,
                image_size,
                palette,
            ),
            "Clean Segmentation",
        ),

        add_title(
            tensor_to_pil(patched_image),
            "Patched Image",
        ),

        add_title(
            segmentation_to_pil(
                patched_prediction,
                image_size,
                palette,
            ),
            "Patched Segmentation",
        ),
    ]

    canvas = Image.new(
        "RGB",
        (
            sum(panel.width for panel in panels),
            max(panel.height for panel in panels),
        ),
        "white",
    )

    x = 0

    for panel in panels:
        canvas.paste(
            panel,
            (x, 0),
        )

        x += panel.width

    canvas.save(save_path)


def evaluate_patch_model(
    patch_model,
    segmentation_model,
    dataloader,
    save_dir,
    patch_device,
    segmentation_device,
    num_visual_samples=8,
):
    """
    Evaluate clean vs patched segmentation.

    If the existing dataset loader returns "target",
    pixel accuracy and mIoU are calculated.

    Otherwise prediction_change_rate is still reported.
    """

    save_dir = Path(save_dir)
    sample_dir = save_dir / "samples"

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    patch_model.eval()
    segmentation_model.eval()

    clean_confusion = None
    patched_confusion = None

    total_changed = 0
    total_prediction_pixels = 0

    num_classes = None
    palette = None
    sample_count = 0
    image_count = 0
    labels_available = None

    with torch.inference_mode():

        progress = tqdm(
            dataloader,
            desc="Evaluation",
            dynamic_ncols=True,
        )

        for batch in progress:
            image_cpu = batch["image"]

            if labels_available is None:
                labels_available = (
                    "target" in batch
                )

            # ==========================================
            # Clean segmentation -> segmentation GPU
            # ==========================================

            clean_image_seg = image_cpu.to(
                segmentation_device,
                non_blocking=True,
            )

            clean_logits = (
                segmentation_model(
                    clean_image_seg
                )
            )

            clean_prediction = (
                clean_logits.argmax(
                    dim=1
                )
            )

            # ==========================================
            # Patch generation -> patch GPU
            # ==========================================

            patch_input = image_cpu.to(
                patch_device,
                non_blocking=True,
            )

            patch, mask = patch_model(
                patch_input
            )

            patched_image = (
                patch_input
                * (1.0 - mask)
                +
                patch
                * mask
            )

            # ==========================================
            # Patched segmentation
            # ==========================================

            patched_logits = (
                segmentation_model(
                    patched_image.to(
                        segmentation_device
                    )
                )
            )

            patched_prediction = (
                patched_logits.argmax(
                    dim=1
                )
            )

            # ==========================================
            # Setup
            # ==========================================

            if num_classes is None:
                num_classes = (
                    clean_logits.shape[1]
                )

                palette = build_palette(
                    num_classes
                )

                if labels_available:
                    clean_confusion = (
                        torch.zeros(
                            (
                                num_classes,
                                num_classes,
                            ),
                            dtype=torch.int64,
                        )
                    )

                    patched_confusion = (
                        torch.zeros_like(
                            clean_confusion
                        )
                    )

            # ==========================================
            # Prediction change rate
            # ==========================================

            if (
                patched_prediction.shape[-2:]
                != clean_prediction.shape[-2:]
            ):
                patched_prediction = (
                    F.interpolate(
                        patched_prediction
                        .unsqueeze(1)
                        .float(),

                        size=(
                            clean_prediction
                            .shape[-2:]
                        ),

                        mode="nearest",
                    )
                    .squeeze(1)
                    .long()
                )

            changed = (
                clean_prediction
                != patched_prediction
            )

            total_changed += (
                changed.sum().item()
            )

            total_prediction_pixels += (
                changed.numel()
            )

            # ==========================================
            # Ground-truth metrics when available
            # ==========================================

            if labels_available:
                target = batch["target"].to(
                    segmentation_device,
                    non_blocking=True,
                )

                clean_target = (
                    resize_target_to_logits(
                        target,
                        clean_logits,
                    )
                )

                patched_target = (
                    resize_target_to_logits(
                        target,
                        patched_logits,
                    )
                )

                clean_confusion = (
                    update_confusion_matrix(
                        clean_confusion,
                        clean_prediction,
                        clean_target,
                        num_classes,
                    )
                )

                patched_confusion = (
                    update_confusion_matrix(
                        patched_confusion,
                        patched_prediction,
                        patched_target,
                        num_classes,
                    )
                )

            # ==========================================
            # Visual samples
            # ==========================================

            for index in range(
                image_cpu.shape[0]
            ):
                if (
                    sample_count
                    >= num_visual_samples
                ):
                    break

                save_comparison(
                    clean_image=(
                        image_cpu[index]
                    ),

                    clean_prediction=(
                        clean_prediction[index]
                    ),

                    patched_image=(
                        patched_image[index]
                        .cpu()
                    ),

                    patched_prediction=(
                        patched_prediction[index]
                    ),

                    palette=palette,

                    save_path=(
                        sample_dir
                        /
                        f"sample_{sample_count:04d}.png"
                    ),
                )

                sample_count += 1

            image_count += (
                image_cpu.shape[0]
            )

            progress.set_postfix({
                "change":
                    (
                        f"{total_changed / max(total_prediction_pixels, 1):.4f}"
                    )
            })

    summary = {
        "num_images":
            image_count,

        "prediction_change_rate":
            (
                total_changed
                /
                max(
                    total_prediction_pixels,
                    1,
                )
            ),
    }

    if labels_available:
        clean_metrics = (
            compute_metrics(
                clean_confusion
            )
        )

        patched_metrics = (
            compute_metrics(
                patched_confusion
            )
        )

        summary.update({
            "clean_pixel_accuracy":
                clean_metrics[
                    "pixel_accuracy"
                ],

            "patched_pixel_accuracy":
                patched_metrics[
                    "pixel_accuracy"
                ],

            "accuracy_drop":
                (
                    clean_metrics[
                        "pixel_accuracy"
                    ]
                    -
                    patched_metrics[
                        "pixel_accuracy"
                    ]
                ),

            "clean_miou":
                clean_metrics[
                    "miou"
                ],

            "patched_miou":
                patched_metrics[
                    "miou"
                ],

            "miou_drop":
                (
                    clean_metrics[
                        "miou"
                    ]
                    -
                    patched_metrics[
                        "miou"
                    ]
                ),
        })

    with open(
        save_dir
        / "evaluation_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    return summary