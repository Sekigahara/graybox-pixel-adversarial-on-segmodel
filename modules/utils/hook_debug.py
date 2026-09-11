import torch

from modules.model.feature_utils.feature_extraction import FeatureHook

def validate_feature_extraction(
    segmentation_model,
    device,
):
    segmentation_model.eval()
    feature_module = (segmentation_model.get_feature_module())
    hook = FeatureHook(feature_module)
    H, W = (segmentation_model.get_input_size())

    image = torch.rand(
        1,
        3,
        H,
        W,
        device=device,
        requires_grad=True,
    )

    hook.clear()

    logits = segmentation_model(
        image
    )

    feature = hook.feature

    if feature is None:
        hook.remove()

        raise RuntimeError(
            "Feature module produced "
            "no captured feature."
        )

    print(
        "Segmentation logits:",
        logits.shape
    )

    print(
        "Semantic feature:",
        feature.shape
    )

    print(
        "Feature requires grad:",
        feature.requires_grad
    )

    if feature.ndim != 4:
        hook.remove()

        raise ValueError(
            "Semantic feature must have "
            "shape [B,C,H,W], but got "
            f"{feature.shape}"
        )

    if not feature.requires_grad:
        hook.remove()

        raise RuntimeError(
            "Captured feature does not "
            "participate in autograd."
        )

    hook.remove()
