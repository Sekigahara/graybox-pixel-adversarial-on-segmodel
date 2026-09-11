import os
import torch
import argparse

from pathlib import Path
from torch.utils.data import DataLoader

from modules.model.seg_model_loader import load_segmentation_model
from modules.dataset.dataset_loader import VistasDataset
from modules.utils.utils import load_yaml

from modules.model.patch_evaluation import (
    load_patch_model,
    evaluate_patch_model,
    evaluate_segmentation_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate normal and patched segmentation results."
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default="config/bench/config_test_bench.yaml",
        help="Path to the YAML configuration file",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["normal", "comparison"],
        default="comparison",
        help=(
            "normal: image -> segmentation only, "
            "comparison: clean and patched results"
        ),
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--min_label_area",
        type=int,
        default=150,
        help="Minimum predicted region area before its label is drawn.",
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
    config = load_yaml(args.config)

    if torch.cuda.device_count() >= 2:
        patch_device = torch.device("cuda:0")
        segmentation_device = torch.device("cuda:1")
    else:
        patch_device = torch.device("cuda:0")
        segmentation_device = patch_device

    print("DEBUG : Loading Segmentation Model")

    seg_model = load_segmentation_model(
        name=config["seg_model"]["name"]
    ).to(segmentation_device)

    seg_model.eval()

    patch_model = None

    if args.mode == "comparison":
        print("DEBUG : Loading Patch Model")

        patch_model = load_patch_model(
            checkpoint_path=config["patch_model"]["ckpt_path"],
            config=config,
            segmentation_model=seg_model,
            device=patch_device,
        )

    print("DEBUG : Loading Validation Dataset")

    val_dataset = VistasDataset(
        root=config["dataset"]["dataset_root_path"],
        split="validation",
        image_size=seg_model.get_input_size(),
    )

    num_workers = config["dataset"]["num_workers"]

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    if args.mode == "comparison":
        checkpoint_path = Path(
            config["patch_model"]["ckpt_path"]
        )

        save_dir = checkpoint_path.parent.parent / "evaluation"
    else:
        normal_save_root = config.get(
            "export",
            {},
        ).get(
            "save_dir",
            "./evaluation",
        )

        save_dir = (
            Path(normal_save_root)
            / "normal_evaluation"
        )

    ignore_index = config["dataset"].get("ignore_index")

    if hasattr(seg_model, "get_ignore_index"):
        ignore_index = seg_model.get_ignore_index()

    print("DEBUG : Starting Evaluation")

    if args.mode == "normal":
        summary = evaluate_segmentation_model(
            segmentation_model=seg_model,
            dataloader=val_loader,
            save_dir=save_dir,
            segmentation_device=segmentation_device,
            num_visual_samples=args.num_samples,
            min_label_area=args.min_label_area,
            ignore_index=ignore_index,
        )

    else:
        summary = evaluate_patch_model(
            patch_model=patch_model,
            segmentation_model=seg_model,
            dataloader=val_loader,
            save_dir=save_dir,
            patch_device=patch_device,
            segmentation_device=segmentation_device,
            num_visual_samples=args.num_samples,
            min_label_area=args.min_label_area,
            ignore_index=ignore_index,
        )

    print("Evaluation Result")

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
