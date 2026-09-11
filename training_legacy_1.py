import os
import torch
import argparse

from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

# Model Modules
from modules.model.adversarial_model.conv_encoder import ConvEncoder
from modules.model.adversarial_model.vit_encoder import ViTEncoder
from modules.model.adversarial_model.decoder import PatchMaskModel
from modules.model.seg_model_loader import load_segmentation_model

# Other than models
from modules.utils.export_import import save_patch_model
from modules.utils.utils import load_yaml
from modules.training.patch_segmentation_trainer import PatchSegmentationTrainer
from modules.dataset.dataset_loader import VistasDataset
from modules.utils.hook_debug import validate_feature_extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the patch and mask generator."
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to the YAML configuration file",
        default="config/training/config_test.yaml",
    )

    parser.add_argument(
        "--gpu_num",
        type=str,
        default="0,1",
    )

    args = parser.parse_args()

    if args.gpu_num != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    return args


def main() -> None:
    args = parse_args()

    # Load configuration
    config = load_yaml(args.config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            "The current training pipeline requires two visible GPUs."
        )

    # CUDA_VISIBLE_DEVICES remaps the selected physical GPUs.
    patch_device = torch.device("cuda:0")
    segmentation_device = torch.device("cuda:1")

    print("DEBUG : Loading Segmentation Model")

    seg_model = load_segmentation_model(
        name=config["seg_model"]["name"]
    ).to(segmentation_device)

    print("DEBUG : Seg Model check on feature reshape")

    validate_feature_extraction(
        seg_model,
        segmentation_device,
    )

    print("DEBUG : Loading Encoder Model")

    if config["encoder_model"]["type"] == "conv":
        encoder_model = ConvEncoder(
            model_name=config["encoder_model"]["name"],
            pretrained=True,
        )

    else:
        encoder_model = ViTEncoder(
            model_name=config["encoder_model"]["name"],
            pretrained=True,
        )

    # Freeze the encoder and unfreeze only the requested
    # final stages/blocks before constructing the trainer.
    encoder_model.set_trainable_last_layers(
        num_layers=config["encoder_model"]["n_trainable_layers"]
    )

    print("DEBUG : Loading Decoder Model")

    patch_model = PatchMaskModel(
        encoder=encoder_model,
        patch_size=seg_model.get_input_size(),
        epsilon=config["training"]["eps"],
        mask_temperature=config["training"]["mask_temperature"],
        selection_grid=config["training"]["selection_grid"],
    )

    print("DEBUG : Loading Dataset")

    train_dataset = VistasDataset(
        root=config["dataset"]["dataset_root_path"],
        split="training",
        image_size=seg_model.get_input_size(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=True,
        persistent_workers=config["dataset"]["num_workers"] > 0,
        drop_last=False,
    )

#     val_dataset = VistasDataset(
#         root=config["dataset"]["dataset_root_path"],
#         split="validation",
#         image_size=seg_model.get_input_size(),
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config["dataset"]["batch_size"],
#         shuffle=False,
#         num_workers=config["dataset"]["num_workers"],
#         pin_memory=True,
#         persistent_workers=config["dataset"]["num_workers"] > 0,
#         drop_last=False,
#     )

    # Kept for the later validation pipeline.
    #_ = val_loader

    base_save_patch = os.path.join(
        config["export"]["save_dir"],
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    print("DEBUG : Trainer initialize")

    adv_trainer = PatchSegmentationTrainer(
        segmentation_model=seg_model,
        patch_model=patch_model,
        dataloader=train_loader,
        epochs=config["training"]["epochs"],
        scheduler_dict=config["training"]["scheduler"],
        weight_decay=config["training"]["weight_decay"],
        loss_weight_dict=config["training"]["loss_weight"],
        ignore_index=seg_model.get_ignore_index(),
        save_dir=base_save_patch,
        checkpoint_interval=(
            config["export"]["checkpoint_interval"]
        ),
    )

    print("DEBUG : Start Training")

    # Close the feature hook
    try:
        training_history = adv_trainer.train()

    finally:
        adv_trainer.close()

    print("DEBUG : Saving Final Patch Model")

    model_dir = os.path.join(
        base_save_patch,
        "model",
    )

    os.makedirs(
        model_dir,
        exist_ok=True,
    )

    save_patch_model(
        adv_trainer.patch_model,
        save_path=os.path.join(
            model_dir,
            "model.pt",
        ),
        encoder_type=config["encoder_model"]["type"],
        encoder_name=config["encoder_model"]["name"],
        patch_size=seg_model.get_input_size(),
        epsilon=config["training"]["eps"],
    )


if __name__ == "__main__":
    main()