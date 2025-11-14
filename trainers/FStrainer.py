"""
NeRF trainer class for modular training pipeline.
"""

from typing import Dict

import torch
import torch.nn.functional as F

from config import NeRFConfig
from utils.render_utils import (
    render_image_with_occgrid,
    generate_camera_rays_with_perturbation,
)
from utils.optimizer_utils import AdamLD
from utils.render_utils import NERF_SYNTHETIC_SCENES
from trainers.trainer import NeRFTrainer


class FSTrainer(NeRFTrainer):
    """
    Modular Few Shot NeRF trainer.
    """

    def __init__(self, config: NeRFConfig):
        super().__init__(config)
        self.start = 0.0
        self.end = 0.80
        self.init_level = 6.5
        if config.scene in NERF_SYNTHETIC_SCENES:
            # for blender dataset
            self.init_level = 4.0

    def _setup_optimizers(self):
        """Initialize optimizers and schedulers."""
        self.optimizer = AdamLD(
            list(self.radiance_field.parameters())[1:],
            lr=self.config.training.learning_rate,
            noise_factor=0.99,
        )

        # Schedulers
        milestones = [
            self.config.training.max_steps // 2,
            self.config.training.max_steps * 3 // 4,
            self.config.training.max_steps * 9 // 10,
        ]

        self.scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.01, total_iters=100
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=milestones, gamma=0.33
                ),
            ]
        )

        # Gradient scaler
        self.grad_scaler = torch.cuda.amp.GradScaler(
            self.config.training.grad_scaler_init
        )

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.

        Returns:
            Dictionary containing loss and metrics for this step
        """
        # Set models to training mode
        self.radiance_field.train()
        self.estimator.train()

        # Calculate progress factor and target level
        progress = self.step / self.config.training.max_steps
        alpha = min(progress / (self.end - self.start), 1.0)
        target_level = self.init_level * (1.0 - alpha)

        def occ_eval_fn(x):
            density = self.radiance_field.query_density(x)
            return density * self.render_step_size

        # Sample training data
        i = torch.randint(0, len(self.train_dataset), (1,)).item()
        data = self.train_dataset[i]

        render_bkgd = data["color_bkgd"]
        pixels = data["pixels"]

        c2w = data["c2w"]
        x = data["x"]
        y = data["y"]

        # Generate rays
        rays = generate_camera_rays_with_perturbation(
            x, y, c2w, self.train_dataset, target_level
        )

        # update occupancy grid
        self.estimator.update_every_n_steps(
            step=self.step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )

        # render
        # NOTE: hard-coded cone alpha
        rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
            self.radiance_field,
            self.estimator,
            rays,
            # rendering options
            near_plane=self.config.scene_config.near_plane,
            render_step_size=self.render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=0.004,
            alpha_thre=0.01,
        )

        white_thr = 0.99
        bkgd_mask = pixels.mean(dim=-1) > white_thr
        tau = 0.01  # set >0 (e.g., 0.01) if you want a small allowed haze
        acc_flat = acc.squeeze(-1)
        occ_pen_hinge = torch.relu(acc_flat - tau)  # pushes acc -> 0
        loss_bkgd = (
            occ_pen_hinge[bkgd_mask].mean()
            if bkgd_mask.any()
            else acc_flat.new_zeros(())
        )

        # Compute MLE loss
        if alpha < 1.0:
            loss = self.mle_loss(rgb, pixels)
        else:
            loss = F.smooth_l1_loss(rgb, pixels)

        loss += 0.01 * loss_bkgd

        for param_group in self.optimizer.param_groups:
            param_group["noise_factor"] = (1.0 - alpha) * 0.99

        # Backward pass
        torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()

        self.optimizer.step()
        self.scheduler.step()

        if self.target_sample_batch_size > 0:
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(
                num_rays
                * max((self.target_sample_batch_size / float(n_rendering_samples)), 1.0)
            )
            self.train_dataset.update_num_rays(num_rays)

        with torch.no_grad():
            # Compute metrics
            mse_loss = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse_loss) / torch.log(torch.tensor(10.0))

            self.step += 1

        return {
            "loss": loss.item(),
            "mse_loss": mse_loss.item(),
            "psnr": psnr.item(),
            "num_rays": len(pixels),
            "max_depth": depth.max().item(),
        }

    def mle_loss(self, rgb, pixels):
        eps = 1e-8
        rgb_render = torch.clamp(rgb, min=eps)
        rgb_gt = torch.clamp(pixels, min=eps)

        loss = torch.mean(-torch.log(rgb_render) * rgb_gt) / torch.mean(rgb_gt)
        loss += torch.log(torch.mean(rgb_render))
        loss += (torch.mean(rgb_render) - torch.mean(rgb_gt)) ** 2

        return loss
