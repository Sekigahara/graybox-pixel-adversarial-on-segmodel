import torch
import torch.nn.functional as F


def segmentation_margin_loss(
    logits,
    target,
    kappa=0.0,
    ignore_index=None,
):
    """
    Untargeted pixel-wise attack loss.

    Minimizing this loss pushes the strongest wrong class
    above the ground-truth class.
    """

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

    safe_target = target.clone()
    safe_target[~valid] = 0

    true_logits = logits.gather(
        dim=1,
        index=safe_target.unsqueeze(1),
    ).squeeze(1)

    top_values, top_indices = torch.topk(
        logits,
        k=2,
        dim=1,
    )

    max_wrong = torch.where(
        top_indices[:, 0] == safe_target,
        top_values[:, 1],
        top_values[:, 0],
    )

    pixel_loss = F.relu(
        true_logits
        - max_wrong
        + kappa
    )

    return pixel_loss[valid].mean()


def normalize_spatial_map(
    spatial_map,
    eps=1e-8,
):
    """
    Normalize every sample independently to [0, 1].
    """

    minimum = spatial_map.amin(
        dim=(2, 3),
        keepdim=True,
    )

    spatial_map = spatial_map - minimum

    maximum = spatial_map.amax(
        dim=(2, 3),
        keepdim=True,
    )

    return spatial_map / (
        maximum + eps
    )


def semantic_importance_map(
    clean_feature,
    feature_gradient,
    output_size=None,
):
    """
    Gradient-weighted semantic activation importance.

    Output:
        [B, 1, H, W]
    """

    importance = (
        clean_feature.detach().abs()
        * feature_gradient.detach().abs()
    )

    importance = importance.mean(
        dim=1,
        keepdim=True,
    )

    importance = normalize_spatial_map(
        importance
    )

    if output_size is not None:
        importance = F.interpolate(
            importance,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

    return importance.detach()


def attack_importance_map(
    dense_delta,
    delta_gradient,
):
    """
    First-order importance of retaining each perturbation.

    A high value means removing that pixel is expected
    to weaken the attack.
    """

    importance = -(
        dense_delta.detach()
        * delta_gradient.detach()
    ).sum(
        dim=1,
        keepdim=True,
    )

    importance = F.relu(
        importance
    )

    return normalize_spatial_map(
        importance
    )


def visibility_map(
    dense_delta,
):
    """
    Per-pixel visual modification cost.
    """

    visibility = dense_delta.detach().abs().mean(
        dim=1,
        keepdim=True,
    )

    return normalize_spatial_map(
        visibility
    )


def build_retention_target(
    dense_delta,
    delta_gradient,
    semantic_importance,
    attack_weight=0.6,
    semantic_weight=0.4,
    visibility_weight=0.25,
):
    """
    Combine attack usefulness, semantic importance,
    and visual cost into one detached retention target.
    """

    attack_importance = attack_importance_map(
        dense_delta,
        delta_gradient,
    )

    if (
        semantic_importance.shape[-2:]
        != dense_delta.shape[-2:]
    ):
        semantic_importance = F.interpolate(
            semantic_importance,
            size=dense_delta.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    semantic_importance = normalize_spatial_map(
        semantic_importance.detach()
    )

    visibility = visibility_map(
        dense_delta
    )

    retention_target = (
        attack_weight * attack_importance
        + semantic_weight * semantic_importance
    )

    retention_target = retention_target / (
        1.0
        + visibility_weight * visibility
    )

    retention_target = normalize_spatial_map(
        retention_target
    )

    return retention_target.detach()


def retention_loss(
    retention_logits,
    retention_target,
):
    """
    Train the retention head to rank influential pixels.

    Cosine alignment is used because Top-K depends mainly
    on spatial ranking rather than absolute score values.
    """

    prediction = torch.sigmoid(
        retention_logits
    ).flatten(
        start_dim=1
    )

    target = retention_target.detach().flatten(
        start_dim=1
    )

    prediction = F.normalize(
        prediction,
        p=2,
        dim=1,
    )

    target = F.normalize(
        target,
        p=2,
        dim=1,
    )

    similarity = (
        prediction * target
    ).sum(
        dim=1
    )

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
    """
    Disrupt semantically important intermediate features,
    with more emphasis on currently retained locations.
    """

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

    clean_feature = F.normalize(
        clean_feature.detach(),
        p=2,
        dim=1,
    )

    adv_feature = F.normalize(
        adv_feature,
        p=2,
        dim=1,
    )

    similarity = (
        clean_feature * adv_feature
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

    return (
        weight * semantic_penalty
    ).sum() / (
        weight.sum() + eps
    )


def reconstruction_loss(
    adversarial_image,
    clean_image,
    eps=1e-3,
):
    """
    Recover unnecessary perturbations toward the clean image.
    """

    difference = (
        adversarial_image
        - clean_image
    )

    return (
        torch.sqrt(
            difference.pow(2)
            + eps ** 2
        )
        - eps
    ).mean()


def gradient_preservation_loss(
    adversarial_image,
    clean_image,
):
    """
    Preserve local image edges and texture transitions.
    """

    clean_dx = (
        clean_image[:, :, :, 1:]
        - clean_image[:, :, :, :-1]
    )

    clean_dy = (
        clean_image[:, :, 1:, :]
        - clean_image[:, :, :-1, :]
    )

    adversarial_dx = (
        adversarial_image[:, :, :, 1:]
        - adversarial_image[:, :, :, :-1]
    )

    adversarial_dy = (
        adversarial_image[:, :, 1:, :]
        - adversarial_image[:, :, :-1, :]
    )

    loss_x = torch.abs(
        adversarial_dx - clean_dx
    ).mean()

    loss_y = torch.abs(
        adversarial_dy - clean_dy
    ).mean()

    return loss_x + loss_y


def stealth_loss(
    adversarial_image,
    clean_image,
    gradient_weight=0.2,
):
    """
    Replacement for the old mask TV loss.

    The reconstruction term restores the image, while the
    gradient term preserves local visual structure.
    """

    reconstruction = reconstruction_loss(
        adversarial_image,
        clean_image,
    )

    gradient = gradient_preservation_loss(
        adversarial_image,
        clean_image,
    )

    total = (
        reconstruction
        + gradient_weight * gradient
    )

    return total, reconstruction, gradient


def budget_loss(
    soft_mask,
    keep_ratio,
):
    """
    Keep the soft pruning mask close to the scheduled ratio.
    The final hard Top-K stage already satisfies the budget exactly.
    """

    current_ratio = soft_mask.mean(
        dim=(1, 2, 3)
    )

    target_ratio = torch.full_like(
        current_ratio,
        float(keep_ratio),
    )

    return F.mse_loss(
        current_ratio,
        target_ratio,
    )


def support_diversity_loss(
    soft_mask,
    grid_size=4,
    min_entropy=0.55,
    eps=1e-8,
):
    """
    Prevent all retained perturbations from collapsing into
    one image corner without forcing an exact uniform grid.

    The loss becomes zero once the coarse regional entropy is
    above min_entropy.
    """

    region_mass = F.adaptive_avg_pool2d(
        soft_mask,
        output_size=(grid_size, grid_size),
    ).flatten(
        start_dim=1
    )

    region_probability = region_mass / (
        region_mass.sum(
            dim=1,
            keepdim=True,
        )
        + eps
    )

    entropy = -(
        region_probability
        * torch.log(
            region_probability + eps
        )
    ).sum(
        dim=1
    )

    maximum_entropy = torch.log(
        soft_mask.new_tensor(
            float(grid_size * grid_size)
        )
    )

    normalized_entropy = entropy / maximum_entropy

    loss = F.relu(
        min_entropy - normalized_entropy
    ).mean()

    return loss, normalized_entropy.mean()