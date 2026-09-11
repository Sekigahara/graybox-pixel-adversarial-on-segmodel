import os
import torch
import argparse

from pathlib import Path
from torch.utils.data import DataLoader

from modules.utils.utils import load_yaml
from modules.model.adversarial_model.conv_encoder import ConvEncoder
from modules.model.adversarial_model.vit_encoder import ViTEncoder
from modules.model.seg_model_loader import load_segmentation_model
from modules.dataset.dataset_loader import VistasDatasetLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and inspect a YAML configuration file."
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file",
        default='config/training/config_test.yaml'
    )
    
    parser.add_argument("--gpu_num", type=str, default='2,3')
    args = parser.parse_args()
    
    # Set GPU Available Devices first
    if args.gpu_num != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
        
    return args

def main() -> None:
    args = parse_args()

    # Load the config
    config = load_yaml(args.config)
    
    print("DEBUG : Loading Segmentation Model")
    # Load segmentation model first
    seg_model = load_segmentation_model(
        name=config['seg_model']['name']
    )
    
    print("DEBUG : Loading Encoder Model")
    # Load based on the encoder type
    if config['encoder_model']['type'] == 'conv':
        encoder_model = ConvEncoder(
            model_name=config['encoder_model']['name']
        )
    else:
        encoder_model = ViTEncoder(
            model_name=config['encoder_model']['name']
        )
    
    print("DEBUG : Loading Dataset")
    # Load the dataset
    train_dataset = VistasDatasetLoader(
        root=config['dataset']['dataset_root_path'],
        split='training',
        image_size=seg_model.get_input_size(),
    )
    val_dataset = VistasDatasetLoader(
        root=config['dataset']['dataset_root_path'],
        split='validation',
        image_size=image_size,
    )
    

if __name__ == "__main__":
    main()