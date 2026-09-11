import torch

import torch.nn.functional as F

def masked_tv_loss(scores):
    tv_h = torch.abs(
        scores[:, :, 1:, :]
        - scores[:, :, :-1, :]
    ).mean()

    tv_w = torch.abs(
        scores[:, :, :, 1:]
        - scores[:, :, :, :-1]
    ).mean()

    return tv_h + tv_w

def segmentation_margin_loss(
    logits,
    target,
    kappa=0.0,
):

    true_logits = logits.gather(
        dim=1,
        index=target.unsqueeze(1),
    ).squeeze(1)

    wrong_logits = logits.clone()

    wrong_logits.scatter_(
        dim=1,
        index=target.unsqueeze(1),
        value=float("-inf"),
    )

    max_wrong = wrong_logits.max(
        dim=1
    ).values

    loss = F.relu(
        true_logits
        - max_wrong
        + kappa
    )

    return loss.mean()

def semantic_feature_loss(
    clean_feature,
    adv_feature,
    feature_gradient,
    rho=0.5,
    eps=1e-8,
):

    # ----------------------------------------
    # Semantic importance
    #
    # activation × gradient
    # ----------------------------------------

    importance = (
        clean_feature.detach().abs()
        *
        feature_gradient.detach().abs()
    )

    importance = importance.mean(
        dim=1
    )

    # [B, H, W]

    # ----------------------------------------
    # Cosine similarity at each location
    # ----------------------------------------

    clean_norm = F.normalize(
        clean_feature.detach(),
        p=2,
        dim=1
    )

    adv_norm = F.normalize(
        adv_feature,
        p=2,
        dim=1
    )

    similarity = (
        clean_norm * adv_norm
    ).sum(
        dim=1
    )

    # [B, H, W]

    # ----------------------------------------
    # Disrupt only until sufficiently different
    # ----------------------------------------

    semantic_penalty = F.relu(
        similarity - rho
    )

    loss = (
        importance
        * semantic_penalty
    ).sum()

    loss = loss / (
        importance.sum()
        + eps
    )

    return loss