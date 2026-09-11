import torch
import torch.nn.functional as F


def segmentation_margin_loss(
    logits,
    target,
    kappa=0.0,
    ignore_index=None,
):
    num_classes = logits.shape[1]

    valid = (
        (target >= 0)
        & (target < num_classes)
    )

    if ignore_index is not None:
        valid = valid & (
            target != ignore_index
        )

    if not valid.any():
        return logits.sum() * 0.0

    # Replace ignored/invalid IDs temporarily so gather()
    # never receives an out-of-range index.
    safe_target = target.clone()
    safe_target[~valid] = 0

    true_logits = logits.gather(
        dim=1,
        index=safe_target.unsqueeze(1),
    ).squeeze(1)

    # Find the strongest wrong class without cloning the
    # complete [B,C,H,W] tensor.
    top_values, top_indices = torch.topk(
        logits,
        k=2,
        dim=1,
    )

    top1_value = top_values[:, 0]
    top2_value = top_values[:, 1]
    top1_class = top_indices[:, 0]

    max_wrong = torch.where(
        top1_class == safe_target,
        top2_value,
        top1_value,
    )

    pixel_loss = F.relu(
        true_logits
        - max_wrong
        + kappa
    )

    return pixel_loss[valid].mean()


def semantic_importance_map(
    clean_feature,
    feature_gradient,
    output_size=None,
    eps=1e-8,
):
    importance = (
        clean_feature.detach().abs()
        * feature_gradient.detach().abs()
    )

    importance = importance.mean(
        dim=1,
        keepdim=True,
    )

    importance = importance / (
        importance.amax(
            dim=(2, 3),
            keepdim=True,
        )
        + eps
    )

    if output_size is not None:
        importance = F.interpolate(
            importance,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

    return importance.detach()


def optimal_location_loss(
    location_scores,
    semantic_importance,
    eps=1e-8,
):
    if (
        semantic_importance.shape[-2:]
        != location_scores.shape[-2:]
    ):
        semantic_importance = F.interpolate(
            semantic_importance,
            size=location_scores.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    semantic_importance = (
        semantic_importance
        / (
            semantic_importance.amax(
                dim=(2, 3),
                keepdim=True,
            )
            + eps
        )
    )

    location_vector = F.normalize(
        location_scores.flatten(1),
        p=2,
        dim=1,
    )

    importance_vector = F.normalize(
        semantic_importance.flatten(1),
        p=2,
        dim=1,
    )

    similarity = (
        location_vector
        * importance_vector
    ).sum(dim=1)

    return (
        1.0 - similarity
    ).mean()


def semantic_feature_loss(
    clean_feature,
    adv_feature,
    feature_gradient,
    soft_mask,
    rho=0.5,
    eps=1e-8,
):
    importance = semantic_importance_map(
        clean_feature,
        feature_gradient,
    )

    selected_locations = F.interpolate(
        soft_mask,
        size=clean_feature.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    clean_norm = F.normalize(
        clean_feature.detach(),
        p=2,
        dim=1,
    )

    adv_norm = F.normalize(
        adv_feature,
        p=2,
        dim=1,
    )

    similarity = (
        clean_norm * adv_norm
    ).sum(
        dim=1,
        keepdim=True,
    )

    semantic_penalty = F.relu(
        similarity - rho
    )

    weight = (
        importance
        * selected_locations
    )

    loss = (
        weight * semantic_penalty
    ).sum() / (
        weight.sum() + eps
    )

    return loss


def color_tv_loss(
    color_delta,
):
    tv_h = torch.abs(
        color_delta[:, :, 1:, :]
        - color_delta[:, :, :-1, :]
    ).mean()

    tv_w = torch.abs(
        color_delta[:, :, :, 1:]
        - color_delta[:, :, :, :-1]
    ).mean()

    return tv_h + tv_w

def local_color_loss(
    image,
    patch,
    soft_mask,
    kernel_size=5,
    eps=1e-8,
):
    padding = kernel_size // 2
    neighborhood_size = (
        kernel_size * kernel_size
    )

    padded_image = F.pad(
        image,
        (
            padding,
            padding,
            padding,
            padding,
        ),
        mode="reflect",
    )

    local_sum = F.avg_pool2d(
        padded_image,
        kernel_size=kernel_size,
        stride=1,
    )

    local_sum = (
        local_sum
        * neighborhood_size
    )

    # Exclude the center pixel.
    local_mean = (
        local_sum - image
    ) / (
        neighborhood_size - 1
    )

    color_difference = torch.abs(
        patch - local_mean
    ).mean(
        dim=1,
        keepdim=True,
    )

    loss = (
        color_difference
        * soft_mask
    ).sum()

    loss = loss / (
        soft_mask.sum()
        + eps
    )

    return loss
