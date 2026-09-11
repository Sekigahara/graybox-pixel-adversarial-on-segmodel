import os
import torch
import argparse

from pathlib import Path
from torch.utils.data import DataLoader

from modules.model.seg_model_loader import (
    load_segmentation_model
)

from modules.dataset.dataset_loader import (
    VistasDataset
)

from modules.utils.utils import load_yaml

from modules.model.patch_evaluation import (
    load_patch_model,
    evaluate_patch_model,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and inspect a YAML configuration file."
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to the YAML configuration file",
        default='config/bench/config_test_bench.yaml'
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10
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
    if torch.cuda.device_count() >= 2:
        patch_device = torch.device("cuda:0")
        segmentation_device = torch.device("cuda:1")
    else:
        patch_device = torch.device("cuda:0")
        segmentation_device = patch_device
    
    # Loading segmodel
    seg_model = load_segmentation_model(
        name=config["seg_model"]["name"]
    ).to(segmentation_device)
    seg_model.eval()
    
    patch_model = load_patch_model(
        checkpoint_path=config["patch_model"]["ckpt_path"],
        config=config,
        segmentation_model=seg_model,
        device=patch_device,
    )
    
    # Validation
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
    
    save_dir = Path(
        os.path.join(
            '/'.join(config["patch_model"]["ckpt_path"].split("/")[0:3]),
            "evaluation"
        )
    )
    
    # Summary and saving
    summary = evaluate_patch_model(
        patch_model=patch_model,
        segmentation_model=seg_model,
        dataloader=val_loader,
        save_dir=save_dir,
        patch_device=patch_device,
        segmentation_device=segmentation_device,
        num_visual_samples=args.num_samples
    )

if __name__ == "__main__":
    main()