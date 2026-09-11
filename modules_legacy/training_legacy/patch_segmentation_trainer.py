import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from pathlib import Path

from modules.utils.logger import TrainingLogger
from modules.model.feature_utils.feature_extraction import FeatureHook
from modules.training.loss_function import segmentation_margin_loss, masked_tv_loss, semantic_feature_loss

class PatchSegmentationTrainer:
    def __init__(
        self,
        segmentation_model,
        patch_model,
        dataloader,
        scheduler_dict:dict,
        epochs: int = 20,
        weight_decay: float = 1e-4,
        loss_weight_dict: dict = None,
        ignore_index=None,
        save_dir: str = "./experiment",
        checkpoint_interval: int = 5,
        device=None,
    ):
        # ==================================================
        # Device
        # ==================================================

        self.patch_device = torch.device("cuda:0")
        self.segmentation_device = torch.device("cuda:1")
        self.scheduler_dict = scheduler_dict
        
#         if device is None:

#             device = torch.device(
#                 "cuda"
#                 if torch.cuda.is_available()
#                 else "cpu"
#             )

#         self.device = device

        # ==================================================
        # Training configuration
        # ==================================================

        self.epochs = epochs
        self.dataloader = dataloader

        self.loss_weight_dict = (
            loss_weight_dict
        )

        self.ignore_index = (
            ignore_index
        )

        self.checkpoint_interval = (
            checkpoint_interval
        )

        # ==================================================
        # Patch / Mask model
        #
        # Encoder trainability was ALREADY configured
        # outside this trainer.
        # ==================================================

        self.patch_model = (
            patch_model.to(
                self.patch_device
            )
        )

        # ==================================================
        # Segmentation model
        #
        # Victim model is always frozen.
        # ==================================================

        self.segmentation_model = (
            segmentation_model.to(
                self.segmentation_device
            )
        )

        self.segmentation_model.eval()

        for parameter in (
            self.segmentation_model.parameters()
        ):

            parameter.requires_grad = False

        # ==================================================
        # Semantic feature module
        #
        # Mandatory because L_sem requires it.
        # ==================================================

        self.feature_module = (
            self.segmentation_model.get_feature_module()
        )

        self.feature_hook = FeatureHook(self.feature_module)

        # ==================================================
        # Optimizer
        #
        # ONLY parameters configured as trainable
        # outside are added.
        # ==================================================

        trainable_parameters = [
            parameter for parameter in self.patch_model.parameters() if parameter.requires_grad
        ]

        if len(trainable_parameters) == 0:
            raise RuntimeError(
                "Patch model contains no "
                "trainable parameters."
            )

        learning_rate = self.scheduler_dict.get(
            "lr",
            0.0001,
        )
        self.optimizer = (
            torch.optim.AdamW(
                trainable_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        )
        
        # Define training scheduler
        warmup_epochs = self.scheduler_dict.get(
            "warmup_epochs",
            0,
        )

        warmup_start_factor = self.scheduler_dict.get(
            "warmup_start_factor",
            0.1,
        )

        min_lr = self.scheduler_dict.get(
            "min_lr",
            1e-6,
        )

        if warmup_epochs < 0:
            raise ValueError(
                "warmup_epochs must be >= 0."
            )

        if warmup_epochs >= self.epochs:
            raise ValueError(
                "warmup_epochs must be smaller "
                "than total epochs."
            )

        # ----------------------------------------------
        # Warmup + Cosine
        # ----------------------------------------------

        if warmup_epochs > 0:
            warmup_scheduler = (
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=warmup_start_factor,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
            )

            cosine_scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=(
                        self.epochs
                        - warmup_epochs
                    ),
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

        # ==================================================
        # Saving
        # ==================================================

        self.save_dir = Path(
            save_dir
        )

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
            mode='append'
        )

        self.best_accuracy_drop = (
            float("-inf")
        )

        self.start_epoch = 1

        # ==================================================
        # Print trainable parameter information
        # ==================================================

        total_params = sum(
            p.numel()
            for p in self.patch_model.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in self.patch_model.parameters()
            if p.requires_grad
        )

        print(
            f"Patch model parameters: "
            f"{total_params:,}"
        )

        print(
            f"Trainable parameters: "
            f"{trainable_params:,}"
        )

        print(
            f"Trainable ratio: "
            f"{100 * trainable_params / total_params:.2f}%"
        )
        
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

        clean_logits = (
            self.segmentation_model(
                clean_image
            )
        )

        clean_feature = (
            self.feature_hook.feature
        )

        if clean_feature is None:
            raise RuntimeError(
                "Failed to capture clean "
                "semantic feature."
            )

        if self.ignore_index is None:
            seg_loss = F.cross_entropy(
                clean_logits,
                target,
            )
        else:
            seg_loss = F.cross_entropy(
                clean_logits,
                target,
                ignore_index=self.ignore_index,
            )

        feature_gradient = (
            torch.autograd.grad(
                outputs=seg_loss,
                inputs=clean_feature,
                create_graph=False,
                retain_graph=False,
            )[0]
        )

        return (
            clean_logits.detach(),
            clean_feature.detach(),
            feature_gradient.detach(),
        )
    
    @torch.no_grad()
    def _pixel_accuracy(
        self,
        logits,
        target
    ):
        prediction = logits.argmax(
            dim=1
        )

        if self.ignore_index is None:
            valid = torch.ones_like(
                target,
                dtype=torch.bool
            )
        else:

            valid = (
                target
                != self.ignore_index
            )

        if valid.sum() == 0:
            return 0.0

        return (
            prediction[valid]
            == target[valid]
        ).float().mean().item()
    
    def train_step(
        self,
        batch,
    ):
        # ==========================================
        # Original batch remains CPU
        # ==========================================

        image_cpu = batch["image"]
        target_cpu = batch["target"]

        # ==========================================
        # GPU 0:
        # Patch / Mask Generator
        # ==========================================

        image_patch = image_cpu.to(
            self.patch_device,
            non_blocking=True
        )

        # ==========================================
        # GPU 1:
        # Clean segmentation reference
        # ==========================================

        clean_image_seg = image_cpu.to(
            self.segmentation_device,
            non_blocking=True
        )

        target_seg = target_cpu.to(
            self.segmentation_device,
            non_blocking=True
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        # ==========================================
        # 1. Semantic teacher
        #
        # Entirely GPU 1
        # ==========================================

        (
            clean_logits,
            clean_feature,
            feature_gradient,
        ) = self._get_semantic_reference(
            clean_image_seg,
            target_seg,
        )

        # ==========================================
        # 2. Patch + mask
        #
        # Entirely GPU 0
        # ==========================================

        patch, mask = self.patch_model(
            image_patch
        )

        # GPU 0
        adv_image = (
            image_patch
            * (1.0 - mask)
            +
            patch
            * mask
        )

        # ==========================================
        # 3. Transfer ONLY generated image
        #
        # GPU 0 -> GPU 1
        #
        # IMPORTANT:
        # Do NOT detach().
        #
        # This copy remains part of autograd.
        # ==========================================

        adv_image_seg = adv_image.to(
            self.segmentation_device
        )

        # ==========================================
        # 4. Adversarial segmentation
        #
        # GPU 1
        # ==========================================

        self.feature_hook.clear()

        adv_logits = (
            self.segmentation_model(
                adv_image_seg
            )
        )

        adv_feature = (
            self.feature_hook.feature
        )

        if adv_feature is None:

            raise RuntimeError(
                "Failed to capture adversarial "
                "semantic feature."
            )

        # ==========================================
        # Attack loss
        #
        # GPU 1
        # ==========================================

        attack_loss = (
            segmentation_margin_loss(
                logits=adv_logits,
                target=target_seg,
                kappa=(
                    self.loss_weight_dict[
                        "kappa"
                    ]
                ),

#                 ignore_index=(
#                     self.ignore_index
#                 ),
            )
        )

        # ==========================================
        # Semantic loss
        #
        # GPU 1
        # ==========================================

        semantic_loss = (
            semantic_feature_loss(
                clean_feature=clean_feature,
                adv_feature=adv_feature,

                feature_gradient=(
                    feature_gradient
                ),

                rho=(
                    self.loss_weight_dict[
                        "rho"
                    ]
                ),
            )
        )

        # ==========================================
        # TV
        #
        # Mask lives on GPU 0.
        # ==========================================

        tv_loss = masked_tv_loss(
            mask
        )

        # ==========================================
        # Total loss must be on ONE GPU.
        #
        # Move scalar TV loss to GPU 1.
        #
        # Do not detach it.
        # ==========================================

        tv_loss_seg = tv_loss.to(
            self.segmentation_device
        )

        total_loss = (
            self.loss_weight_dict["l_atk"]
            * attack_loss

            +

            self.loss_weight_dict["l_sem"]
            * semantic_loss

            +

            self.loss_weight_dict["l_tv"]
            * tv_loss_seg
        )

        # ==========================================
        # Backward:
        #
        # GPU1 loss
        #   ↓
        # segmentation model
        #   ↓
        # GPU1 → GPU0 copy
        #   ↓
        # PatchMaskModel
        #
        # Seg model has no parameter gradients.
        # ==========================================

        total_loss.backward()

        self.optimizer.step()

        # ==========================================
        # Metrics all on GPU 1
        # ==========================================

        clean_accuracy = (
            self._pixel_accuracy(
                clean_logits,
                target_seg,
            )
        )

        adv_accuracy = (
            self._pixel_accuracy(
                adv_logits.detach(),
                target_seg,
            )
        )

        return {
            "loss":
                total_loss.item(),

            "attack_loss":
                attack_loss.item(),

            "semantic_loss":
                semantic_loss.item(),

            "tv_loss":
                tv_loss.item(),

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
        # ==================================================
        # Victim segmentation model always eval
        # ==================================================

        self.segmentation_model.eval()

        # ==================================================
        # Decoder in train mode
        # ==================================================

        self.patch_model.mask_decoder.train()

        # ==================================================
        # Keep encoder mode according to whatever
        # freezing / fine-tuning configuration you
        # already applied outside the trainer
        # ==================================================

        if hasattr(
            self.patch_model.encoder,
            "set_finetune_mode"
        ):
            self.patch_model.encoder.set_finetune_mode()

        # ==================================================
        # Metric accumulators
        # ==================================================

        totals = {
            "loss": 0.0,
            "attack_loss": 0.0,
            "semantic_loss": 0.0,
            "tv_loss": 0.0,
            "clean_accuracy": 0.0,
            "adv_accuracy": 0.0,
            "accuracy_drop": 0.0,
        }

        num_batches = 0

        # ==================================================
        # Progress bar
        # ==================================================

        progress_bar = tqdm(
            self.dataloader,
            desc=(
                f"Epoch "
                f"{epoch:03d}/{self.epochs:03d}"
            ),
            leave=True,
            dynamic_ncols=True,
        )

        # ==================================================
        # Training batches
        # ==================================================

        for batch in progress_bar:
            result = self.train_step(
                batch
            )

            num_batches += 1

            for key in totals:

                totals[key] += (
                    result[key]
                )

            # ==============================================
            # Running averages
            # ==============================================

            running_loss = (
                totals["loss"]
                / num_batches
            )

            running_atk = (
                totals["attack_loss"]
                / num_batches
            )

            running_sem = (
                totals["semantic_loss"]
                / num_batches
            )

            running_tv = (
                totals["tv_loss"]
                / num_batches
            )

            running_clean_acc = (
                totals["clean_accuracy"]
                / num_batches
            )

            running_adv_acc = (
                totals["adv_accuracy"]
                / num_batches
            )

            running_drop = (
                totals["accuracy_drop"]
                / num_batches
            )

            # ==============================================
            # Update tqdm postfix
            # ==============================================

            progress_bar.set_postfix({
                "loss":
                    f"{running_loss:.4f}",

                "atk":
                    f"{running_atk:.4f}",

                "sem":
                    f"{running_sem:.4f}",

                "tv":
                    f"{running_tv:.4f}",

                "clean":
                    f"{running_clean_acc:.3f}",

                "adv":
                    f"{running_adv_acc:.3f}",

                "drop":
                    f"{running_drop:.3f}",
            })

        if num_batches == 0:

            raise RuntimeError(
                "Dataloader produced no batches."
            )

        # ==================================================
        # Epoch averages
        # ==================================================

        metrics = {
            key:
                value / num_batches

            for key, value
            in totals.items()
        }

        metrics["epoch"] = epoch

        metrics["learning_rate"] = (
            self.optimizer
            .param_groups[0]["lr"]
        )

        return metrics
        
    def train(self):
        for epoch in range(self.epochs):
            # Do feedforward and backprop
            metrics = self.train_epoch(epoch)
            # Log all the trained metrics
            self.logger.log(metrics)
        
    def save_checkpoint(
        self,
        epoch,
        metrics,
        filename,
        additional_savename
    ):
        checkpoint = {
            "epoch":
                epoch,

            # Only patch generator needs saving.
            "patch_model":
                self.patch_model.state_dict(),

            "optimizer":
                self.optimizer.state_dict(),

            "loss_weight_dict":
                self.loss_weight_dict,

            "metrics":
                metrics,

            "best_accuracy_drop":
                self.best_accuracy_drop,
        }

        path = (
            self.checkpoint_dir
            / filename
        )

        torch.save(
            checkpoint,
            path
        )

        return path