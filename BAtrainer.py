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
)

from trainer import NeRFTrainer

from utils.lie_utils import LIE_
from utils.pose_utils import POSE_


class BATrainer(NeRFTrainer):
    """
    Bundle Adjustment NeRF trainer with proposal networks.
    """

    def __init__(self, config: NeRFConfig):
        super().__init__(config)
        self.se3_noise_factor = 0.10

    def _setup_models(self):
        super()._setup_models()

        # Add noise to camera poses for bundle adjustment
        se3_noise = (
            torch.randn(len(self.train_dataset), 6, device=self.device)
            * self.se3_noise_factor
        )
        self.se3_noise_pose = LIE_.se3_to_SE3(se3_noise)

        # pose refinement embeddings
        self.se3_refine = torch.nn.Embedding(len(self.train_data), 6).to(self.device)
        torch.nn.init.zeros_(self.se3_refine.weight)

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
        c2w_perturbed_pose = POSE_.compose(se3_noise_pose, c2w_pose)  # [:,3,4]
        c2w_refined_pose = POSE_.compose(
            LIE_.se3_to_SE3(self.se3_refine.weight), c2w_perturbed_pose
        )  # [:,3,4]

        c2w = POSE_.to_matrix(c2w_refined_pose)  # [:,4,4]

        # Generate rays
        rays = generate_camera_rays(x, y, c2w, self.train_dataset)

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
        loss = F.smooth_l1_loss(rgb, pixels)

        # Backward pass
        self.optimizer.zero_grad()
        self.pose_optimizer.zero_grad()

        self.grad_scaler.scale(loss).backward()

        self.optimizer.step()
        self.scheduler.step()

        self.pose_optimizer.step()
        self.pose_scheduler.step()

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

    def print_training_stats(self, metrics: Dict[str, float]):
        """Print training statistics."""
        elapsed_time = time.time() - self.start_time
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={self.step} | "
            f"loss={metrics['loss']:.5f} | psnr={metrics['psnr']:.2f} | "
            f"num_rays={metrics['num_rays']:d} | "
            f"max_depth={metrics['max_depth']:.3f}"
        )
