import torch
from pathlib import Path

def save_patch_model(
    model,
    save_path,
    encoder_type,
    encoder_name,
    patch_size,
    epsilon,
):
    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "encoder_type":
            encoder_type,

        "encoder_name":
            encoder_name,

        "patch_size":
            tuple(patch_size),

        "epsilon":
            epsilon,
    }

    torch.save(
        checkpoint,
        save_path,
    )

    print(
        f"Patch model saved to: "
        f"{save_path}"
    )


def load_patch_model(
    checkpoint_path,
    device=None,
    pretrained_encoder=False,
):
    """
    Load a saved PatchMaskModel.

    pretrained_encoder=False is normally enough
    because the checkpoint already contains the
    encoder weights.
    """

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    encoder_type = (
        checkpoint["encoder_type"]
    )

    encoder_name = (
        checkpoint["encoder_name"]
    )

    patch_size = tuple(
        checkpoint["patch_size"]
    )

    epsilon = (
        checkpoint["epsilon"]
    )

    # ==========================================
    # Rebuild encoder
    # ==========================================

    if encoder_type == "conv":

        encoder = ConvEncoder(
            model_name=encoder_name,
            pretrained=pretrained_encoder,
        )

    elif encoder_type == "vit":

        encoder = ViTEncoder(
            model_name=encoder_name,
            pretrained=pretrained_encoder,
        )

    else:

        raise ValueError(
            f"Unknown encoder type: "
            f"{encoder_type}"
        )

    # ==========================================
    # Rebuild PatchMaskModel
    # ==========================================

    model = PatchMaskModel(
        encoder=encoder,
        patch_size=patch_size,
        epsilon=epsilon,
    )

    # ==========================================
    # Restore weights
    # ==========================================

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    return model

def save_training_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    save_path,
    encoder_type,
    encoder_name,
    patch_size,
    epsilon,
    loss_weight_dict,
):
    checkpoint = {
        # Model
        "model_state_dict":
            model.state_dict(),

        # Optimizer
        "optimizer_state_dict":
            optimizer.state_dict(),

        # Training state
        "epoch":
            epoch,

        "metrics":
            metrics,

        # Architecture
        "encoder_type":
            encoder_type,

        "encoder_name":
            encoder_name,

        "patch_size":
            tuple(patch_size),

        "epsilon":
            epsilon,

        # Loss configuration
        "loss_weight_dict":
            loss_weight_dict,
    }

    torch.save(
        checkpoint,
        save_path,
    )

def load_training_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    start_epoch = (
        checkpoint["epoch"]
        + 1
    )

    metrics = checkpoint.get(
        "metrics",
        {}
    )

    return (
        model,
        optimizer,
        start_epoch,
        metrics,
    )
