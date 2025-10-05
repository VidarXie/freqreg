"""
NeRF trainer class for modular training pipeline.
"""

from typing import Dict

import torch
import torch.nn.functional as F

from config import NeRFConfig
from utils.render_utils import (
    render_image_with_propnet,
    generate_camera_rays_with_perturbation,
)

from trainer import NeRFTrainer


class MLETrainer(NeRFTrainer):
    """
    Modular NeRF trainer with proposal networks.
    """

    def __init__(self, config: NeRFConfig):
        super().__init__(config)
        self.start = 0.0
        self.end = 0.75
        self.init_level = 4.0

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.

        Returns:
            Dictionary containing loss and metrics for this step
        """
        # Set models to training mode
        self.radiance_field.train()
        for p in self.proposal_networks:
            p.train()
        self.estimator.train()

        # Calculate progress factor and target level
        progress = self.step / self.config.training.max_steps
        alpha = min(progress / (self.end - self.start), 1.0)
        target_level = self.init_level * (1.0 - alpha)

        # Sample training data
        i = torch.randint(0, len(self.train_dataset), (1,)).item()
        data = self.train_dataset[i]

        """
        contains:
            "pixels": pixels,  
            "color_bkgd": color_bkgd,  
            "c2w": c2w,
            "image_id": image_id,
            "x": x,
            "y": y,
        """

        render_bkgd = data["color_bkgd"]
        pixels = data["pixels"]

        c2w = data["c2w"]
        image_id = data["image_id"]
        x = data["x"]
        y = data["y"]

        # Generate rays
        rays = generate_camera_rays_with_perturbation(
            x, y, c2w, self.train_dataset, target_level
        )

        # Determine if proposal networks need gradients
        proposal_requires_grad = self.proposal_requires_grad_fn(self.step)

        # Render
        rgb, acc, depth, extras = render_image_with_propnet(
            self.radiance_field,
            self.proposal_networks,
            self.estimator,
            rays,
            # rendering options
            num_samples=self.config.model.num_samples,
            num_samples_per_prop=self.config.model.num_samples_per_prop,
            near_plane=self.config.scene_config.near_plane,
            far_plane=self.config.scene_config.far_plane,
            sampling_type=self.config.model.sampling_type,
            opaque_bkgd=self.config.model.opaque_bkgd,
            render_bkgd=render_bkgd,
            # train options
            proposal_requires_grad=proposal_requires_grad,
        )

        # Update estimator
        self.estimator.update_every_n_steps(
            extras["trans"], proposal_requires_grad, loss_scaler=1024
        )

        # Compute MLE loss
        if alpha < 1.0:
            loss = self.mle_loss(rgb, pixels)
        else:
            loss = F.smooth_l1_loss(rgb, pixels)

        # Backward pass
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.optimizer.step()
        self.scheduler.step()

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
