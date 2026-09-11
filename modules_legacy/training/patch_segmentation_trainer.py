from pathlib import Path

import torch
import torch.nn.functional as F

from tqdm.auto import tqdm

from modules.utils.logger import TrainingLogger
from modules.model.feature_utils.feature_extraction import FeatureHook
from modules.training.loss_function import (
    segmentation_margin_loss,
    semantic_importance_map,
    optimal_location_loss,
    semantic_feature_loss,
    color_tv_loss,
    local_color_loss,
)


class PatchSegmentationTrainer:
    def __init__(
        self,
        segmentation_model,
        patch_model,
        dataloader,
        scheduler_dict,
        epochs=20,
        weight_decay=1e-4,
        loss_weight_dict=None,
        ignore_index=None,
        save_dir="./experiment",
        checkpoint_interval=5,
        log_mode="overwrite",
        patch_device="cuda:0",
        segmentation_device="cuda:1",
    ):
        self.patch_device = torch.device(
            patch_device
        )

        self.segmentation_device = torch.device(
            segmentation_device
        )

        self.epochs = epochs
        self.dataloader = dataloader
        self.scheduler_dict = scheduler_dict
        self.loss_weight_dict = loss_weight_dict
        self.ignore_index = ignore_index
        self.checkpoint_interval = checkpoint_interval

        self.patch_model = patch_model.to(
            self.patch_device
        )

        self.segmentation_model = segmentation_model.to(
            self.segmentation_device
        )

        self.segmentation_model.eval()

        for parameter in self.segmentation_model.parameters():
            parameter.requires_grad = False

        self.feature_module = (
            self.segmentation_model
            .get_feature_module()
        )

        self.feature_hook = FeatureHook(
            self.feature_module
        )

        trainable_parameters = [
            parameter
            for parameter
            in self.patch_model.parameters()
            if parameter.requires_grad
        ]

        learning_rate = self.scheduler_dict.get("lr", 1e-4)

        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        warmup_epochs = self.scheduler_dict.get(
            "warmup_epochs",
            0,
        )

        min_lr = self.scheduler_dict.get(
            "min_lr",
            1e-6,
        )

        if warmup_epochs > 0:
            warmup_scheduler = (
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=self.scheduler_dict.get(
                        "warmup_start_factor",
                        0.1,
                    ),
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
            )

            cosine_scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.epochs - warmup_epochs,
                    eta_min=min_lr,
                )
            )

            self.scheduler = (
                torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[
                        warmup_scheduler,
                        cosine_scheduler,
                    ],
                    milestones=[
                        warmup_epochs
                    ],
                )
            )
        else:
            self.scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.epochs,
                    eta_min=min_lr,
                )
            )

        self.save_dir = Path(
            save_dir
        )

        self.checkpoint_dir = (
            self.save_dir
            / "checkpoints"
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.logger = TrainingLogger(
            self.save_dir,
            mode=log_mode,
        )

        self.start_epoch = 1
        self.best_accuracy_drop = float(
            "-inf"
        )

    def _resize_target(
        self,
        target,
        logits,
    ):
        if (
            target.shape[-2:]
            == logits.shape[-2:]
        ):
            return target

        target = F.interpolate(
            target.unsqueeze(1).float(),
            size=logits.shape[-2:],
            mode="nearest",
        )

        return target.squeeze(1).long()

    def _get_semantic_reference(
        self,
        image,
        target,
    ):
        clean_image = (
            image
            .detach()
            .requires_grad_(True)
        )

        self.feature_hook.clear()

        clean_logits = self.segmentation_model(
            clean_image
        )

        clean_feature = (
            self.feature_hook.feature
        )

        target_native = self._resize_target(
            target,
            clean_logits,
        )

        clean_loss = F.cross_entropy(
            clean_logits,
            target_native,
            ignore_index=self.ignore_index,
        )

        feature_gradient = torch.autograd.grad(
            outputs=clean_loss,
            inputs=clean_feature,
            create_graph=False,
            retain_graph=False,
        )[0]

        semantic_importance = (
            semantic_importance_map(
                clean_feature,
                feature_gradient,
                output_size=image.shape[-2:],
            )
        )

        return (
            clean_logits.detach(),
            clean_feature.detach(),
            feature_gradient.detach(),
            semantic_importance,
        )

    @torch.no_grad()
    def _pixel_accuracy(
        self,
        logits,
        target,
    ):
        target = self._resize_target(
            target,
            logits,
        )

        prediction = logits.argmax(
            dim=1
        )

        valid = (
            (target >= 0)
            & (target < logits.shape[1])
        )

        if self.ignore_index is not None:
            valid = valid & (
                target != self.ignore_index
            )

        if not valid.any():
            return 0.0

        return (
            prediction[valid]
            == target[valid]
        ).float().mean().item()

    def train_step(
        self,
        batch,
    ):
        image_cpu = batch["image"]
        target_cpu = batch["target"]

        image_patch = image_cpu.to(
            self.patch_device,
            non_blocking=True,
        )

        clean_image_seg = image_cpu.to(
            self.segmentation_device,
            non_blocking=True,
        )

        target_seg = target_cpu.to(
            self.segmentation_device,
            non_blocking=True,
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        (
            clean_logits,
            clean_feature,
            feature_gradient,
            semantic_importance,
        ) = self._get_semantic_reference(
            clean_image_seg,
            target_seg,
        )

        patch_output = self.patch_model(
            image_patch,
            return_aux=True,
        )

        patch = patch_output["patch"]
        mask = patch_output["mask"]
        soft_mask = patch_output["soft_mask"]

        location_scores = patch_output[
            "location_scores"
        ]

        color_delta = patch_output[
            "color_delta"
        ]

        adv_image = patch_output[
            "adversarial_image"
        ]

        # Calculate location loss
        location_loss = optimal_location_loss(
            location_scores=location_scores,

            semantic_importance=(
                semantic_importance.to(
                    self.patch_device
                )
            ),
        )
        
        # Calculate camouflage loss
        camouflage_loss = local_color_loss(
            image=image_patch,
            patch=patch,
            soft_mask=soft_mask,

            kernel_size=(
                self.loss_weight_dict.get(
                    "camouflage_kernel",
                    5,
                )
            ),
        )
        
        # TV Loss
        tv_loss = color_tv_loss(
            color_delta
        )

        # Clear the hook mem
        self.feature_hook.clear()
        
        # Adversarial segmentation
        adv_logits = self.segmentation_model(
            adv_image.to(
                self.segmentation_device
            )
        )

        adv_feature = (
            self.feature_hook.feature
        )

        target_native = self._resize_target(
            target_seg,
            adv_logits,
        )
        
        attack_loss = segmentation_margin_loss(
            logits=adv_logits,
            target=target_native,
            kappa=self.loss_weight_dict[
                "kappa"
            ],
            ignore_index=self.ignore_index,
        )

        semantic_loss = semantic_feature_loss(
            clean_feature=clean_feature,
            adv_feature=adv_feature,
            feature_gradient=feature_gradient,
            soft_mask=soft_mask.to(
                self.segmentation_device
            ),
            rho=self.loss_weight_dict[
                "rho"
            ],
        )

        total_loss = (
            self.loss_weight_dict["l_atk"]
            * attack_loss

            +

            self.loss_weight_dict["l_loc"]
            * location_loss.to(
                self.segmentation_device
            )

            +

            self.loss_weight_dict["l_sem"]
            * semantic_loss

            +

            self.loss_weight_dict["l_tv"]
            * tv_loss.to(
                self.segmentation_device
            )

            +

            self.loss_weight_dict["l_cam"]
            * camouflage_loss.to(
                self.segmentation_device
            )
        )

        total_loss.backward()
        self.optimizer.step()

        clean_accuracy = self._pixel_accuracy(
            clean_logits,
            target_seg,
        )

        adv_accuracy = self._pixel_accuracy(
            adv_logits.detach(),
            target_seg,
        )

        return {
            "loss":
                total_loss.item(),

            "attack_loss":
                attack_loss.item(),

            "location_loss":
                location_loss.item(),

            "semantic_loss":
                semantic_loss.item(),

            "tv_loss":
                tv_loss.item(),

            "camouflage_loss":
                camouflage_loss.item(),

            "clean_accuracy":
                clean_accuracy,

            "adv_accuracy":
                adv_accuracy,

            "accuracy_drop":
                clean_accuracy
                - adv_accuracy,
        }

    def train_epoch(
        self,
        epoch,
    ):
        self.segmentation_model.eval()
        self.patch_model.mask_decoder.train()

        if hasattr(
            self.patch_model.encoder,
            "set_finetune_mode",
        ):
            self.patch_model.encoder.set_finetune_mode()

        totals = {
            "loss": 0.0,
            "attack_loss": 0.0,
            "location_loss": 0.0,
            "semantic_loss": 0.0,
            "tv_loss": 0.0,
            "camouflage_loss": 0.0,
            "clean_accuracy": 0.0,
            "adv_accuracy": 0.0,
            "accuracy_drop": 0.0,
        }

        progress = tqdm(
            self.dataloader,
            desc=(
                f"Epoch "
                f"{epoch:03d}/{self.epochs:03d}"
            ),
            dynamic_ncols=True,
        )

        num_batches = 0

        for batch in progress:
            result = self.train_step(
                batch
            )

            num_batches += 1

            for key in totals:
                totals[key] += result[key]

            running = {
                key:
                    value / num_batches

                for key, value
                in totals.items()
            }

            #print(running.keys())
            progress.set_postfix({
                "loss":
                    f"{running['loss']:.4f}",

                "atk":
                    f"{running['attack_loss']:.4f}",

                "loc":
                    f"{running['location_loss']:.4f}",

                "sem":
                    f"{running['semantic_loss']:.4f}",

                "tv":
                    f"{running['tv_loss']:.4f}",

                "cam":
                    f"{running['camouflage_loss']:.3f}",

                "drop":
                    f"{running['accuracy_drop']:.3f}",
            })

        metrics = {
            key:
                value / num_batches

            for key, value
            in totals.items()
        }

        metrics["epoch"] = epoch
        metrics["learning_rate"] = (
            self.optimizer.param_groups[0]["lr"]
        )

        return metrics

    def save_checkpoint(
        self,
        epoch,
        metrics,
        filename,
    ):
        checkpoint = {
            "epoch":
                epoch,

            "patch_model":
                self.patch_model.state_dict(),

            "optimizer":
                self.optimizer.state_dict(),

            "scheduler":
                self.scheduler.state_dict(),

            "loss_weight_dict":
                self.loss_weight_dict,

            "metrics":
                metrics,

            "best_accuracy_drop":
                self.best_accuracy_drop,
        }

        torch.save(
            checkpoint,
            self.checkpoint_dir
            / filename,
        )


    def load_checkpoint(
        self,
        checkpoint_path,
    ):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        self.patch_model.load_state_dict(
            checkpoint["patch_model"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        self.best_accuracy_drop = checkpoint.get(
            "best_accuracy_drop",
            float("-inf"),
        )

        self.start_epoch = (
            checkpoint["epoch"]
            + 1
        )

    def train(
        self,
    ):
        for epoch in range(
            self.start_epoch,
            self.epochs + 1,
        ):
            metrics = self.train_epoch(epoch)

            self.logger.log(metrics)

            self.scheduler.step()

            is_best = (
                metrics["accuracy_drop"]
                >
                self.best_accuracy_drop
            )

            if is_best:
                self.best_accuracy_drop = (
                    metrics[
                        "accuracy_drop"
                    ]
                )

            self.save_checkpoint(
                epoch,
                metrics,
                "last.pt",
            )

            if (
                self.checkpoint_interval > 0
                and
                epoch
                % self.checkpoint_interval
                == 0
            ):
                self.save_checkpoint(
                    epoch,
                    metrics,
                    f"epoch_{epoch:03d}.pt",
                )

            if is_best:
                self.save_checkpoint(
                    epoch,
                    metrics,
                    "best.pt",
                )

        #return self.logger.history

    def close(
        self,
    ):
        self.feature_hook.remove()