"""
NeRF trainer class for modular training pipeline.
"""

import time
from typing import Dict

import torch
import torch.nn.functional as F

from config import NeRFConfig
from utils.render_utils import (
    render_image_with_propnet,
    generate_camera_rays,
    generate_camera_rays_with_perturbation
)

from trainer import NeRFTrainer

from utils.lie_utils import LIE_
from utils.pose_utils import POSE_, MulPose


class BATrainer(NeRFTrainer):
    """
    Bundle Adjustment NeRF trainer with proposal networks.
    """

    def __init__(self, config: NeRFConfig):
        self.se3_noise_factor = 0.10
        super().__init__(config)
        self.start = 0.0
        self.end = 0.75
        self.init_level = 4.0

    def _setup_models(self):
        super()._setup_models()

        # Add noise to camera poses for bundle adjustment
        se3_noise = (
            torch.randn(len(self.train_dataset), 6, device=self.device)
            * self.se3_noise_factor
        )
        self.se3_noise_pose = LIE_.se3_to_SE3(se3_noise)

        # pose refinement embeddings
        self.se3_refine = torch.nn.Parameter(torch.zeros(len(self.train_dataset), 6, dtype=torch.float).cuda().requires_grad_(True))
        # torch.nn.Embedding(len(self.train_dataset), 6).to(self.device)
        # torch.nn.init.zeros_(self.se3_refine.weight)

    def _setup_optimizers(self):
        """Initialize optimizers and schedulers."""
        super()._setup_optimizers()

        self.pose_optimizer = torch.optim.Adam([self.se3_refine], lr=0.001)

        # Schedulers
        pose_opt_milestones = [
            self.config.training.max_steps // 2,
            self.config.training.max_steps * 3 // 4,
            self.config.training.max_steps * 5 // 6,
            self.config.training.max_steps * 9 // 10,
        ]

        self.pose_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.pose_optimizer,
            milestones=pose_opt_milestones,
            gamma=0.33,
        )

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

        # Add noise to camera pose
        c2w_pose = POSE_.from_matrix(c2w)  # [:,3,4]
        se3_noise_pose = self.se3_noise_pose[image_id]  # [:,3,4]
        c2w_perturbed_pose = POSE_.compose_pair(c2w_pose, se3_noise_pose)
        # MulPose(se3_noise_pose) @ MulPose(c2w_pose)  # [:,3,4]
        c2w_refined_pose = POSE_.compose_pair(
            c2w_perturbed_pose,
            LIE_.se3_to_SE3(self.se3_refine)[image_id])
        # MulPose(LIE_.se3_to_SE3(self.se3_refine)[image_id]) @ MulPose(c2w_perturbed_pose)
        self.se3_refine.retain_grad()
        bottom_row = POSE_.to_matrix(c2w_refined_pose)  # [:,4,4]]
        matrix_4x4 = torch.cat([c2w_refined_pose, bottom_row], dim=-2)

        # Generate rays
        rays = generate_camera_rays_with_perturbation(
            x, y, matrix_4x4, self.train_dataset, target_level
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
        # Compute loss
        if alpha < 1.0:
            loss = self.mle_loss(rgb, pixels)
        else:
            loss = F.smooth_l1_loss(rgb, pixels)

        # Backward pass
        torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad(set_to_none=True)
        self.pose_optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()

        self.optimizer.step()
        self.scheduler.step()

        self.pose_optimizer.step()
        self.pose_scheduler.step()
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
    
    def print_training_stats(self, metrics: Dict[str, float]):
        """Print training statistics."""
        elapsed_time = time.time() - self.start_time
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={self.step} | "
            f"loss={metrics['loss']:.5f} | psnr={metrics['psnr']:.2f} | "
            f"num_rays={metrics['num_rays']:d} | "
            f"max_depth={metrics['max_depth']:.3f}"
        )

        
