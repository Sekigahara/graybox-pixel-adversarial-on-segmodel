import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from modules.utils.logger import TrainingLogger
from modules.model.feature_utils.feature_extraction import FeatureHook
from modules.training.loss_function import (
    segmentation_margin_loss,
    semantic_importance_map,
    build_retention_target,
    retention_loss,
    semantic_feature_loss,
    stealth_loss,
    budget_loss,
    support_diversity_loss,
)

class PatchSegmentationTrainer:
    def __init__(
        self,
        segmentation_model,
        patch_model,
        dataloader,
        scheduler_dict,
        pruning_dict,
        epochs=20,
        weight_decay=1e-4,
        loss_weight_dict=None,
        ignore_index=None,
        save_dir="./experiment",
        checkpoint_interval=5,
        log_mode="overwrite",
        grad_clip=1.0,
        patch_device="cuda:0",
        segmentation_device="cuda:1",
    ):
        self.patch_device = torch.device(patch_device)
        self.segmentation_device = torch.device(segmentation_device)

        self.epochs = epochs
        self.dataloader = dataloader
        self.scheduler_dict = scheduler_dict
        self.pruning_dict = pruning_dict
        self.loss_weight_dict = loss_weight_dict
        self.ignore_index = ignore_index
        self.checkpoint_interval = checkpoint_interval
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay

        self.patch_model = patch_model.to(self.patch_device)
        self.segmentation_model = segmentation_model.to(
            self.segmentation_device
        )

        self.segmentation_model.eval()

        for parameter in self.segmentation_model.parameters():
            parameter.requires_grad = False

        self.feature_module = (
            self.segmentation_model.get_feature_module()
        )

        self.feature_hook = FeatureHook(
            self.feature_module
        )

        trainable_parameters = [
            parameter
            for parameter in self.patch_model.parameters()
            if parameter.requires_grad
        ]

        learning_rate = self.scheduler_dict.get(
            "lr",
            1e-4,
        )

        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = self._build_scheduler()

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(
            parents=True,
            exist_ok=True,
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
        self.best_accuracy_drop = float("-inf")
        self.recovery_started = False

    def _build_scheduler(self):
        warmup_epochs = self.scheduler_dict.get(
            "warmup_epochs",
            0,
        )

        min_lr = self.scheduler_dict.get(
            "min_lr",
            1e-6,
        )

        if warmup_epochs == 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=min_lr,
            )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=self.scheduler_dict.get(
                "warmup_start_factor",
                0.1,
            ),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        cosine_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - warmup_epochs,
                eta_min=min_lr,
            )
        )

        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                warmup_scheduler,
                cosine_scheduler,
            ],
            milestones=[
                warmup_epochs
            ],
        )

    def _get_training_stage(self, epoch):
        dense_epochs = self.pruning_dict[
            "dense_warmup_epochs"
        ]

        pruning_epochs = self.pruning_dict[
            "pruning_epochs"
        ]

        temperature_start = self.pruning_dict.get(
            "mask_temperature_start",
            1.0,
        )

        temperature_end = self.pruning_dict.get(
            "mask_temperature_end",
            self.patch_model.mask_temperature,
        )

        if epoch <= dense_epochs:
            return (
                "dense",
                1.0,
                False,
                self.pruning_dict.get(
                    "dense_stealth_factor",
                    0.0,
                ),
                temperature_start,
            )

        if epoch <= dense_epochs + pruning_epochs:
            progress = (
                epoch - dense_epochs
            ) / pruning_epochs

            keep_ratio = (
                self.patch_model.epsilon
                + (
                    1.0
                    - self.patch_model.epsilon
                )
                * 0.5
                * (
                    1.0
                    + math.cos(
                        math.pi * progress
                    )
                )
            )

            temperature = (
                temperature_start
                + progress
                * (
                    temperature_end
                    - temperature_start
                )
            )

            # Exact scheduled support in the forward pass,
            # soft surrogate in the backward pass.
            return (
                "pruning",
                keep_ratio,
                True,
                progress,
                temperature,
            )

        recovery_epochs = max(
            self.epochs
            - dense_epochs
            - pruning_epochs,
            1,
        )

        recovery_epoch = (
            epoch
            - dense_epochs
            - pruning_epochs
        )

        recovery_progress = (
            recovery_epoch - 1
        ) / max(
            recovery_epochs - 1,
            1,
        )

        stealth_start = self.pruning_dict.get(
            "recovery_stealth_start",
            0.25,
        )

        stealth_factor = (
            stealth_start
            + recovery_progress
            * (
                1.0
                - stealth_start
            )
        )

        return (
            "recovery",
            self.patch_model.epsilon,
            True,
            stealth_factor,
            temperature_end,
        )

    def _start_recovery_stage(self):
        if self.recovery_started:
            return

        self.recovery_started = True

        # Lock the learned support. Only the color-change head
        # is optimized during fixed-support recovery.
        for parameter in self.patch_model.parameters():
            parameter.requires_grad = False

        for parameter in (
            self.patch_model
            .mask_decoder
            .delta_head
            .parameters()
        ):
            parameter.requires_grad = True

        recovery_lr = self.pruning_dict.get(
            "recovery_lr",
            3e-4,
        )

        recovery_min_lr = self.pruning_dict.get(
            "recovery_min_lr",
            1e-5,
        )

        dense_epochs = self.pruning_dict[
            "dense_warmup_epochs"
        ]

        pruning_epochs = self.pruning_dict[
            "pruning_epochs"
        ]

        recovery_epochs = max(
            self.epochs
            - dense_epochs
            - pruning_epochs,
            1,
        )

        self.optimizer = torch.optim.AdamW(
            self.patch_model
            .mask_decoder
            .delta_head
            .parameters(),
            lr=recovery_lr,
            weight_decay=self.weight_decay,
        )

        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=recovery_epochs,
                eta_min=recovery_min_lr,
            )
        )

        print(
            "Starting fixed-support recovery | "
            f"ratio={self.patch_model.epsilon:.4f} | "
            f"lr={recovery_lr:.2e}"
        )

    def _set_model_mode(self, stage):
        self.segmentation_model.eval()

        if stage == "recovery":
            self.patch_model.eval()
            self.patch_model.mask_decoder.delta_head.train()
            return

        self.patch_model.mask_decoder.train()

        if hasattr(
            self.patch_model.encoder,
            "set_finetune_mode",
        ):
            self.patch_model.encoder.set_finetune_mode()

    def _resize_target(
        self,
        target,
        logits,
    ):
        if target.shape[-2:] == logits.shape[-2:]:
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
        clean_image = image.detach().requires_grad_(
            True
        )

        self.feature_hook.clear()

        clean_logits = self.segmentation_model(
            clean_image
        )

        clean_feature = self.feature_hook.feature

        target_native = self._resize_target(
            target,
            clean_logits,
        )

        if self.ignore_index is None:
            clean_loss = F.cross_entropy(
                clean_logits,
                target_native,
            )
        else:
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

        semantic_importance = semantic_importance_map(
            clean_feature,
            feature_gradient,
            output_size=image.shape[-2:],
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
        stage,
        keep_ratio,
        use_hard_mask,
        stealth_factor,
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
            keep_ratio=keep_ratio,
            use_hard_mask=use_hard_mask,
            return_aux=True,
        )

        dense_delta = patch_output[
            "dense_delta"
        ]

        retention_logits = patch_output[
            "retention_logits"
        ]

        soft_mask = patch_output[
            "soft_mask"
        ]

        hard_mask = patch_output[
            "hard_mask"
        ]

        mask = patch_output[
            "mask"
        ]

        adversarial_image = patch_output[
            "adversarial_image"
        ]

        self.feature_hook.clear()

        adv_image_seg = adversarial_image.to(
            self.segmentation_device
        )

        adv_logits = self.segmentation_model(
            adv_image_seg
        )

        adv_feature = self.feature_hook.feature

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

        clean_target = self._resize_target(
            target_seg,
            clean_logits,
        )

        with torch.no_grad():
            clean_attack_loss = segmentation_margin_loss(
                logits=clean_logits,
                target=clean_target,
                kappa=self.loss_weight_dict[
                    "kappa"
                ],
                ignore_index=self.ignore_index,
            )

        attack_gain = (
            clean_attack_loss
            - attack_loss.detach()
        )

        if stage == "recovery":
            keep_loss = dense_delta.new_tensor(0.0)
            scheduled_budget_loss = dense_delta.new_tensor(0.0)
            diversity_loss = dense_delta.new_tensor(0.0)
            support_entropy = dense_delta.new_tensor(0.0)

        else:
            attack_image_gradient = torch.autograd.grad(
                outputs=attack_loss,
                inputs=adv_image_seg,
                retain_graph=True,
                create_graph=False,
            )[0]

            retention_target = build_retention_target(
                dense_delta=dense_delta,
                delta_gradient=attack_image_gradient.detach().to(
                    self.patch_device
                ),
                semantic_importance=semantic_importance.to(
                    self.patch_device
                ),
                attack_weight=self.loss_weight_dict.get(
                    "importance_attack",
                    0.6,
                ),
                semantic_weight=self.loss_weight_dict.get(
                    "importance_semantic",
                    0.4,
                ),
                visibility_weight=self.loss_weight_dict.get(
                    "importance_visibility",
                    0.25,
                ),
            )

            keep_loss = retention_loss(
                retention_logits=retention_logits,
                retention_target=retention_target,
            )

            if keep_ratio < 1.0:
                scheduled_budget_loss = budget_loss(
                    soft_mask=soft_mask,
                    keep_ratio=keep_ratio,
                )

                diversity_loss, support_entropy = (
                    support_diversity_loss(
                        soft_mask=mask,
                        grid_size=self.loss_weight_dict.get(
                            "diversity_grid",
                            4,
                        ),
                        min_entropy=self.loss_weight_dict.get(
                            "min_support_entropy",
                            0.55,
                        ),
                    )
                )
            else:
                scheduled_budget_loss = dense_delta.new_tensor(0.0)
                diversity_loss = dense_delta.new_tensor(0.0)
                support_entropy = dense_delta.new_tensor(1.0)

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

        (
            stealth_total,
            reconstruction,
            gradient_preservation,
        ) = stealth_loss(
            adversarial_image=adversarial_image,
            clean_image=image_patch,
            gradient_weight=self.loss_weight_dict.get(
                "stealth_gradient",
                0.2,
            ),
        )

        keep_weight = self.loss_weight_dict[
            "l_keep"
        ]

        budget_weight = self.loss_weight_dict[
            "l_budget"
        ]

        diversity_weight = self.loss_weight_dict.get(
            "l_diversity",
            0.05,
        )

        if stage == "recovery":
            keep_weight = 0.0
            budget_weight = 0.0
            diversity_weight = 0.0

        total_loss = (
            self.loss_weight_dict["l_atk"]
            * attack_loss
            + keep_weight
            * keep_loss.to(
                self.segmentation_device
            )
            + self.loss_weight_dict["l_sem"]
            * semantic_loss
            + stealth_factor
            * self.loss_weight_dict["l_stealth"]
            * stealth_total.to(
                self.segmentation_device
            )
            + budget_weight
            * scheduled_budget_loss.to(
                self.segmentation_device
            )
            + diversity_weight
            * diversity_loss.to(
                self.segmentation_device
            )
        )

        total_loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in self.patch_model.parameters()
                    if parameter.requires_grad
                ],
                self.grad_clip,
            )

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
            "loss": total_loss.item(),
            "attack_loss": attack_loss.item(),
            "clean_attack_loss": clean_attack_loss.item(),
            "attack_gain": attack_gain.item(),
            "retention_loss": keep_loss.item(),
            "semantic_loss": semantic_loss.item(),
            "stealth_loss": stealth_total.item(),
            "reconstruction_loss": reconstruction.item(),
            "gradient_loss": gradient_preservation.item(),
            "budget_loss": scheduled_budget_loss.item(),
            "diversity_loss": diversity_loss.item(),
            "support_entropy": support_entropy.item(),
            "clean_accuracy": clean_accuracy,
            "adv_accuracy": adv_accuracy,
            "accuracy_drop": clean_accuracy - adv_accuracy,
            "mask_ratio": mask.detach().mean().item(),
            "hard_mask_ratio": hard_mask.detach().mean().item(),
            "soft_mask_ratio": soft_mask.detach().mean().item(),
        }

    def train_epoch(
        self,
        epoch,
    ):
        (
            stage,
            keep_ratio,
            use_hard_mask,
            stealth_factor,
            temperature,
        ) = self._get_training_stage(
            epoch
        )

        if stage == "recovery":
            self._start_recovery_stage()

        self.patch_model.mask_temperature = temperature
        self._set_model_mode(stage)

        totals = {
            "loss": 0.0,
            "attack_loss": 0.0,
            "clean_attack_loss": 0.0,
            "attack_gain": 0.0,
            "retention_loss": 0.0,
            "semantic_loss": 0.0,
            "stealth_loss": 0.0,
            "reconstruction_loss": 0.0,
            "gradient_loss": 0.0,
            "budget_loss": 0.0,
            "diversity_loss": 0.0,
            "support_entropy": 0.0,
            "clean_accuracy": 0.0,
            "adv_accuracy": 0.0,
            "accuracy_drop": 0.0,
            "mask_ratio": 0.0,
            "hard_mask_ratio": 0.0,
            "soft_mask_ratio": 0.0,
        }

        progress_bar = tqdm(
            self.dataloader,
            desc=(
                f"Epoch {epoch:03d}/{self.epochs:03d} "
                f"[{stage}]"
            ),
            dynamic_ncols=True,
        )

        num_batches = 0
        for batch in progress_bar:
            result = self.train_step(
                batch,
                stage=stage,
                keep_ratio=keep_ratio,
                use_hard_mask=use_hard_mask,
                stealth_factor=stealth_factor,
            )

            num_batches += 1

            for key in totals:
                totals[key] += result[key]

            running = {
                key: value / num_batches
                for key, value in totals.items()
            }

            progress_bar.set_postfix({
                "atk": f"{running['attack_loss']:.4f}",
                "gain": f"{running['attack_gain']:.4f}",
                "keep": f"{running['retention_loss']:.4f}",
                "sem": f"{running['semantic_loss']:.4f}",
                "ent": f"{running['support_entropy']:.3f}",
                "drop": f"{running['accuracy_drop']:.3f}",
                "hard": f"{running['hard_mask_ratio']:.3f}",
            })

        metrics = {
            key: value / num_batches
            for key, value in totals.items()
        }

        metrics["epoch"] = epoch
        metrics["stage"] = stage
        metrics["keep_ratio"] = keep_ratio
        metrics["mask_temperature"] = temperature
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
            "epoch": epoch,
            "patch_model": self.patch_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss_weight_dict": self.loss_weight_dict,
            "pruning_dict": self.pruning_dict,
            "metrics": metrics,
            "best_accuracy_drop": self.best_accuracy_drop,
            "recovery_started": self.recovery_started,
        }

        torch.save(
            checkpoint,
            self.checkpoint_dir / filename,
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

        checkpoint_stage = checkpoint.get(
            "metrics",
            {},
        ).get(
            "stage"
        )

        if checkpoint_stage == "recovery":
            self._start_recovery_stage()

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
            checkpoint["epoch"] + 1
        )

    def train(self):
        for epoch in range(
            self.start_epoch,
            self.epochs + 1,
        ):
            metrics = self.train_epoch(
                epoch
            )

            self.logger.log(
                metrics
            )

            self.scheduler.step()

            is_recovery = (
                metrics["stage"] == "recovery"
            )

            is_best = (
                is_recovery
                and metrics["accuracy_drop"]
                > self.best_accuracy_drop
            )

            if is_best:
                self.best_accuracy_drop = (
                    metrics["accuracy_drop"]
                )

            self.save_checkpoint(
                epoch,
                metrics,
                "last.pt",
            )

            if (
                self.checkpoint_interval > 0
                and epoch
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

        return self.logger.history


    def close(self):
        self.feature_hook.remove()
