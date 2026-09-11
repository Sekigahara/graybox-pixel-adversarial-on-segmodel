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
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    encoder_type = checkpoint.get(
        "encoder_type",
        config["encoder_model"]["type"],
    )

    encoder_name = checkpoint.get(
        "encoder_name",
        config["encoder_model"]["name"],
    )

    patch_size = checkpoint.get(
        "patch_size",
        segmentation_model.get_input_size(),
    )

    epsilon = checkpoint.get(
        "epsilon",
        config["patch_model"]["eps"],
    )

    mask_temperature = checkpoint.get(
        "mask_temperature",
        config["patch_model"].get(
            "mask_temperature",
            0.10,
        ),
    )

    if encoder_type == "conv":
        encoder = ConvEncoder(
            model_name=encoder_name,
            pretrained=False,
        )
    else:
        encoder = ViTEncoder(
            model_name=encoder_name,
            pretrained=False,
        )

    patch_model = PatchMaskModel(
        encoder=encoder,
        patch_size=tuple(patch_size),
        epsilon=epsilon,
        mask_temperature=mask_temperature,
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


def get_class_names(
    segmentation_model,
):
    if hasattr(segmentation_model, "get_class_names"):
        class_names = segmentation_model.get_class_names()

    else:
        class_names = segmentation_model.model.config.id2label

    return {
        int(class_id): str(label)
        for class_id, label in class_names.items()
    }


def resize_target_to_logits(
    target,
    logits,
):
    if target.shape[-2:] == logits.shape[-2:]:
        return target

    target = F.interpolate(
        target.unsqueeze(1).float(),
        size=logits.shape[-2:],
        mode="nearest",
    )

    return target.squeeze(1).long()


def resize_prediction(
    prediction,
    image_size,
):
    prediction = F.interpolate(
        prediction[None, None].float(),
        size=image_size,
        mode="nearest",
    )

    return prediction.squeeze().long()


def update_confusion_matrix(
    confusion,
    prediction,
    target,
    num_classes,
    ignore_index=None,
):
    valid = (
        (target >= 0)
        & (target < num_classes)
    )

    if ignore_index is not None:
        valid = valid & (
            target != ignore_index
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
        count.reshape(
            num_classes,
            num_classes,
        ).cpu()
    )

    return confusion


def compute_metrics(
    confusion,
):
    confusion = confusion.float()

    intersection = torch.diag(
        confusion
    )

    pixel_accuracy = (
        intersection.sum()
        / confusion.sum().clamp_min(1)
    )

    union = (
        confusion.sum(dim=1)
        + confusion.sum(dim=0)
        - intersection
    )

    valid = union > 0

    class_iou = torch.zeros_like(
        union
    )

    class_iou[valid] = (
        intersection[valid]
        / union[valid]
    )

    miou = (
        class_iou[valid].mean()
        if valid.any()
        else torch.tensor(0.0)
    )

    return {
        "pixel_accuracy": pixel_accuracy.item(),
        "miou": miou.item(),
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

    for index in range(num_classes):
        hue = (hue + ratio) % 1.0

        rgb = colorsys.hsv_to_rgb(
            hue,
            0.70,
            0.95,
        )

        palette[index] = (
            np.asarray(rgb) * 255
        ).astype(np.uint8)

    if num_classes > 0:
        palette[0] = [0, 0, 0]

    return palette


def tensor_to_pil(
    image,
):
    image = image.detach().cpu().clamp(
        0,
        1,
    )

    image = image.permute(
        1,
        2,
        0,
    ).numpy()

    image = (
        image * 255
    ).round().astype(np.uint8)

    return Image.fromarray(image)


def segmentation_to_pil(
    prediction,
    image_size,
    palette,
    class_names,
    min_label_area,
):
    prediction = resize_prediction(
        prediction,
        image_size,
    ).cpu().numpy()

    prediction = np.clip(
        prediction,
        0,
        len(palette) - 1,
    )

    segmentation = Image.fromarray(
        palette[prediction]
    )

    draw = ImageDraw.Draw(
        segmentation
    )

    printed_labels = []

    for class_id in np.unique(prediction):
        region = prediction == class_id
        area = int(region.sum())

        if area < min_label_area:
            continue

        coordinates = np.argwhere(
            region
        )

        center = coordinates.mean(
            axis=0
        )

        nearest_index = np.argmin(
            (
                (
                    coordinates
                    - center
                ) ** 2
            ).sum(
                axis=1
            )
        )

        y, x = coordinates[
            nearest_index
        ]

        label = class_names.get(
            int(class_id),
            f"class_{class_id}",
        )

        text = f"{class_id}: {label}"

        box = draw.textbbox(
            (0, 0),
            text,
        )

        text_width = box[2] - box[0]
        text_height = box[3] - box[1]

        x = int(
            np.clip(
                x - text_width // 2,
                2,
                segmentation.width
                - text_width
                - 4,
            )
        )

        y = int(
            np.clip(
                y - text_height // 2,
                2,
                segmentation.height
                - text_height
                - 4,
            )
        )

        draw.rectangle(
            (
                x - 2,
                y - 2,
                x + text_width + 2,
                y + text_height + 2,
            ),
            fill="black",
        )

        draw.text(
            (x, y),
            text,
            fill="white",
        )

        printed_labels.append({
            "id": int(class_id),
            "label": label,
            "pixels": area,
        })

    return segmentation, printed_labels


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

    draw = ImageDraw.Draw(panel)

    draw.text(
        (8, 8),
        title,
        fill="black",
    )

    return panel


def join_panels(
    panels,
    save_path,
):
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


def save_normal_result(
    image,
    prediction,
    palette,
    class_names,
    min_label_area,
    save_path,
):
    image_size = image.shape[-2:]

    segmentation, labels = segmentation_to_pil(
        prediction,
        image_size,
        palette,
        class_names,
        min_label_area,
    )

    panels = [
        add_title(
            tensor_to_pil(image),
            "Normal Image",
        ),
        add_title(
            segmentation,
            "Segmentation",
        ),
    ]

    join_panels(
        panels,
        save_path,
    )

    return labels


def save_comparison(
    clean_image,
    clean_prediction,
    patched_image,
    patched_prediction,
    palette,
    class_names,
    min_label_area,
    save_path,
):
    image_size = clean_image.shape[-2:]

    clean_segmentation, clean_labels = segmentation_to_pil(
        clean_prediction,
        image_size,
        palette,
        class_names,
        min_label_area,
    )

    patched_segmentation, patched_labels = segmentation_to_pil(
        patched_prediction,
        image_size,
        palette,
        class_names,
        min_label_area,
    )

    panels = [
        add_title(
            tensor_to_pil(clean_image),
            "Clean Image",
        ),
        add_title(
            clean_segmentation,
            "Clean Segmentation",
        ),
        add_title(
            tensor_to_pil(patched_image),
            "Patched Image",
        ),
        add_title(
            patched_segmentation,
            "Patched Segmentation",
        ),
    ]

    join_panels(
        panels,
        save_path,
    )

    return clean_labels, patched_labels


def evaluate_segmentation_model(
    segmentation_model,
    dataloader,
    save_dir,
    segmentation_device,
    num_visual_samples=8,
    min_label_area=200,
    ignore_index=None,
):
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

    segmentation_model.eval()

    confusion = None
    sample_count = 0
    image_count = 0
    sample_labels = {}

    with torch.inference_mode():
        progress = tqdm(
            dataloader,
            desc="Normal Evaluation",
            dynamic_ncols=True,
        )

        for batch in progress:
            image_cpu = batch["image"]

            image = image_cpu.to(
                segmentation_device,
                non_blocking=True,
            )

            logits = segmentation_model(
                image
            )

            prediction = logits.argmax(
                dim=1
            )

            if confusion is None:
                num_classes = logits.shape[1]
                palette = build_palette(
                    num_classes
                )

                class_names = get_class_names(
                    segmentation_model
                )

                confusion = torch.zeros(
                    (
                        num_classes,
                        num_classes,
                    ),
                    dtype=torch.int64,
                )

            target = batch["target"].to(
                segmentation_device,
                non_blocking=True,
            )

            target = resize_target_to_logits(
                target,
                logits,
            )

            confusion = update_confusion_matrix(
                confusion,
                prediction,
                target,
                num_classes,
                ignore_index,
            )

            for index in range(
                image_cpu.shape[0]
            ):
                if sample_count >= num_visual_samples:
                    break

                labels = save_normal_result(
                    image=image_cpu[index],
                    prediction=prediction[index],
                    palette=palette,
                    class_names=class_names,
                    min_label_area=min_label_area,
                    save_path=(
                        sample_dir
                        / f"sample_{sample_count:04d}.png"
                    ),
                )

                sample_labels[
                    f"sample_{sample_count:04d}"
                ] = labels

                label_text = ", ".join(
                    item["label"]
                    for item in labels
                )

                tqdm.write(
                    f"sample_{sample_count:04d}: "
                    f"{label_text}"
                )

                sample_count += 1

            image_count += image_cpu.shape[0]

    metrics = compute_metrics(
        confusion
    )

    summary = {
        "num_images": image_count,
        "pixel_accuracy": metrics["pixel_accuracy"],
        "miou": metrics["miou"],
    }

    with open(
        save_dir / "evaluation_summary.json",
        "w",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    with open(
        save_dir / "sample_labels.json",
        "w",
    ) as file:
        json.dump(
            sample_labels,
            file,
            indent=4,
        )

    return summary


def evaluate_patch_model(
    patch_model,
    segmentation_model,
    dataloader,
    save_dir,
    patch_device,
    segmentation_device,
    num_visual_samples=8,
    min_label_area=200,
    ignore_index=None,
):
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

    sample_count = 0
    image_count = 0
    sample_labels = {}

    with torch.inference_mode():
        progress = tqdm(
            dataloader,
            desc="Patch Evaluation",
            dynamic_ncols=True,
        )

        for batch in progress:
            image_cpu = batch["image"]

            clean_image = image_cpu.to(
                segmentation_device,
                non_blocking=True,
            )

            clean_logits = segmentation_model(
                clean_image
            )

            clean_prediction = clean_logits.argmax(
                dim=1
            )

            patch_input = image_cpu.to(
                patch_device,
                non_blocking=True,
            )

            patch, mask = patch_model(
                patch_input
            )

            patched_image = (
                patch_input * (1.0 - mask)
                + patch * mask
            )

            patched_logits = segmentation_model(
                patched_image.to(
                    segmentation_device
                )
            )

            patched_prediction = patched_logits.argmax(
                dim=1
            )

            if clean_confusion is None:
                num_classes = clean_logits.shape[1]
                palette = build_palette(
                    num_classes
                )

                class_names = get_class_names(
                    segmentation_model
                )

                clean_confusion = torch.zeros(
                    (
                        num_classes,
                        num_classes,
                    ),
                    dtype=torch.int64,
                )

                patched_confusion = torch.zeros_like(
                    clean_confusion
                )

            if (
                patched_prediction.shape[-2:]
                != clean_prediction.shape[-2:]
            ):
                patched_prediction = F.interpolate(
                    patched_prediction.unsqueeze(1).float(),
                    size=clean_prediction.shape[-2:],
                    mode="nearest",
                )

                patched_prediction = (
                    patched_prediction.squeeze(1).long()
                )

            changed = (
                clean_prediction
                != patched_prediction
            )

            total_changed += changed.sum().item()
            total_prediction_pixels += changed.numel()

            target = batch["target"].to(
                segmentation_device,
                non_blocking=True,
            )

            clean_target = resize_target_to_logits(
                target,
                clean_logits,
            )

            patched_target = resize_target_to_logits(
                target,
                patched_logits,
            )

            clean_confusion = update_confusion_matrix(
                clean_confusion,
                clean_prediction,
                clean_target,
                num_classes,
                ignore_index,
            )

            patched_confusion = update_confusion_matrix(
                patched_confusion,
                patched_prediction,
                patched_target,
                num_classes,
                ignore_index,
            )

            for index in range(
                image_cpu.shape[0]
            ):
                if sample_count >= num_visual_samples:
                    break

                clean_labels, patched_labels = save_comparison(
                    clean_image=image_cpu[index],
                    clean_prediction=clean_prediction[index],
                    patched_image=patched_image[index].cpu(),
                    patched_prediction=patched_prediction[index],
                    palette=palette,
                    class_names=class_names,
                    min_label_area=min_label_area,
                    save_path=(
                        sample_dir
                        / f"sample_{sample_count:04d}.png"
                    ),
                )

                sample_labels[
                    f"sample_{sample_count:04d}"
                ] = {
                    "clean": clean_labels,
                    "patched": patched_labels,
                }

                clean_text = ", ".join(
                    item["label"]
                    for item in clean_labels
                )

                patched_text = ", ".join(
                    item["label"]
                    for item in patched_labels
                )

                tqdm.write(
                    f"sample_{sample_count:04d} clean: "
                    f"{clean_text}"
                )

                tqdm.write(
                    f"sample_{sample_count:04d} patched: "
                    f"{patched_text}"
                )

                sample_count += 1

            image_count += image_cpu.shape[0]

            change_rate = (
                total_changed
                / max(
                    total_prediction_pixels,
                    1,
                )
            )

            progress.set_postfix({
                "change": f"{change_rate:.4f}"
            })

    clean_metrics = compute_metrics(
        clean_confusion
    )

    patched_metrics = compute_metrics(
        patched_confusion
    )

    summary = {
        "num_images": image_count,

        "prediction_change_rate":
            total_changed
            / max(
                total_prediction_pixels,
                1,
            ),

        "clean_pixel_accuracy":
            clean_metrics["pixel_accuracy"],

        "patched_pixel_accuracy":
            patched_metrics["pixel_accuracy"],

        "accuracy_drop":
            clean_metrics["pixel_accuracy"]
            - patched_metrics["pixel_accuracy"],

        "clean_miou":
            clean_metrics["miou"],

        "patched_miou":
            patched_metrics["miou"],

        "miou_drop":
            clean_metrics["miou"]
            - patched_metrics["miou"],
    }

    with open(
        save_dir / "evaluation_summary.json",
        "w",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    with open(
        save_dir / "sample_labels.json",
        "w",
    ) as file:
        json.dump(
            sample_labels,
            file,
            indent=4,
        )

    return summary
