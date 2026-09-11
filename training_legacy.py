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

# Other that models
from modules.utils.export_import import save_patch_model, save_training_checkpoint
from modules.utils.utils import load_yaml
from modules.training.patch_segmentation_trainer import PatchSegmentationTrainer
from modules.dataset.dataset_loader import VistasDataset
from modules.utils.hook_debug import validate_feature_extraction

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and inspect a YAML configuration file."
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to the YAML configuration file",
        default='config/training/config_test.yaml'
    )

    parser.add_argument("--gpu_num", type=str, default='0,1')
    args = parser.parse_args()
    
    # Set GPU Available Devices first
    if args.gpu_num != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
        
    return args

def main() -> None:
    args = parse_args()

    # Load the config
    config = load_yaml(args.config)

    # Get the device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("DEBUG : Loading Segmentation Model")
    # Load segmentation model first
    seg_model = load_segmentation_model(
        name=config['seg_model']['name']
    ).to(device)
    print("DEBUG : Seg Model check on feature reshape")
    # Check whether segmentation model is reshapeable
    validate_feature_extraction(seg_model, device)
    
    print("DEBUG : Loading Encoder Model")
    # Load based on the encoder type
    if config['encoder_model']['type'] == 'conv':
        encoder_model = ConvEncoder(
            model_name=config['encoder_model']['name'],
            pretrained=True
        ).to(device)
    else:
        encoder_model = ViTEncoder(
            model_name=config['encoder_model']['name'],
            pretrained=True
        ).to(device)
    
    # Freeze and set n-trainable layers
    encoder_model.set_trainable_last_layers(
        num_layers=config['encoder_model']['n_trainable_layers']
    )
        
    # Load the Decoder
    print("DEBUG : Loading Decoder Model")
    patch_model = PatchMaskModel(
        encoder=encoder_model,
        patch_size=seg_model.get_input_size(),
        epsilon=config['training']['eps']
    ).to(device)
    
    print("DEBUG : Loading Dataset")
    # Load the dataset
    train_dataset = VistasDataset(
        root=config['dataset']['dataset_root_path'],
        split='training',
        image_size=seg_model.get_input_size(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True
        
    )
    val_dataset = VistasDataset(
        root=config['dataset']['dataset_root_path'],
        split='validation',
        image_size=seg_model.get_input_size(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        pin_memory=True
        
    )
    
    # Saving
    base_save_patch = os.path.join(
        config['export']['save_dir'], datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    print("DEBUG : Trainer initialize")
    # Initialize the trainer
    adv_trainer = PatchSegmentationTrainer(
        segmentation_model=seg_model,
        patch_model=patch_model,
        dataloader=train_loader,
        epochs=config['training']['epochs'],
        scheduler_dict=config['training']['scheduler'],
        weight_decay=config['training']['weight_decay'],
        device=device,
        save_dir=base_save_patch,
        loss_weight_dict=config['training']['loss_weight'],
        checkpoint_interval=config['export']['checkpoint_interval']
    )
    
    # Training
    adv_trainer.train()

    save_patch_model(
        adv_trainer.patch_model,
        save_path=os.path.join(base_save_patch, "model", "model.pt"),
        encoder_type=config['encoder_model']['type'],
        encoder_name=config['encoder_model']['name'],
        patch_size=seg_model.get_input_size(),
        epsilon=config['training']['eps']
    )

if __name__ == "__main__":
    main()