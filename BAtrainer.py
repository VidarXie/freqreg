"""
NeRF trainer class for modular training pipeline.
"""

import time
from typing import Dict
import tqdm

import torch
import torch.nn.functional as F

from config import NeRFConfig
from utils.render_utils import (
    render_image_with_occgrid,
    generate_camera_rays_with_perturbation,
    generate_camera_rays
)

from trainer import NeRFTrainer, NGPRadianceField
from evaluator import NeRFEvaluator

from utils.lie_utils import LIE_
from utils.pose_utils import POSE_, sim3_align_errors
from nerfacc.estimators.occ_grid import OccGridEstimator


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
        """Initialize radiance field and proposal networks."""
        aabb = self.config.to_torch_aabb()

        # NOTE: hard-coded grid resolution and levels
        grid_resolution = (128, 128, 128)
        grid_nlvl = 4
        self.render_step_size = 5e-3
        self.estimator = OccGridEstimator(
            roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
        ).to(self.device)
        # Create main radiance field
        self.radiance_field = NGPRadianceField(
            aabb=aabb, unbounded=self.config.scene_config.unbounded
        ).to(self.device)

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
        self.optimizer = torch.optim.Adam(
            self.radiance_field.parameters(),
            lr=self.config.training.learning_rate,
            eps=self.config.training.eps,
            weight_decay=self.config.training.weight_decay,
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

    def get_pose(self, c2w, noise, refine):
        # Add noise to camera pose
        c2w_pose = POSE_.from_matrix(c2w)  # [:,3,4]
        c2w_perturbed_pose = POSE_.compose_pair(c2w_pose, noise)
        c2w_refined_pose = POSE_.compose_pair(
            c2w_perturbed_pose,
            LIE_.se3_to_SE3(refine))
        return c2w_refined_pose
        
    def get_pose_by_camera(self):
        c2w = self.train_dataset.camtoworlds
        return self.get_pose(c2w, self.se3_noise_pose, self.se3_refine)

    @torch.no_grad()
    def get_pose_error(self):
        est_poses = self.get_pose_by_camera()
        te, re, R0, s, t = sim3_align_errors(self.train_dataset.camtoworlds[..., :3, :4], est_poses)
        return te, re
    
    @torch.no_grad()
    def get_pose_align(self):
        est_poses = self.get_pose_by_camera()
        te, re, R0, s, t = sim3_align_errors(self.train_dataset.camtoworlds[..., :3, :4], est_poses)
        return R0, s, t

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.

        Returns:
            Dictionary containing loss and metrics for this step
        """
        # Set models to training mode
        self.radiance_field.train()
        self.estimator.train()
        def occ_eval_fn(x):
            density = self.radiance_field.query_density(x)
            return density * self.render_step_size
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


        c2w_refined_pose = self.get_pose(c2w, self.se3_noise_pose[image_id], self.se3_refine[image_id])
        matrix_4x4 = POSE_.to_matrix(c2w_refined_pose)  # [:,4,4]]

        # Generate rays
        rays = generate_camera_rays_with_perturbation(
            x, y, matrix_4x4, self.train_dataset, target_level
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
            f"max_depth={metrics['max_depth']:.3f} | "
            f"translation_error={metrics['translation_error']:.3f} | "
            f"rotation_error={metrics['rotation_error']:.3f}"
        )

        
class BAEvaluator(NeRFEvaluator):
    def evaluate(self, verbose = True):
        self.trainer.radiance_field.eval()
        self.trainer.estimator.eval()

        psnrs = []
        lpips_scores = []

        iterator = (
            tqdm.tqdm(range(len(self.trainer.test_dataset)), desc="Evaluating")
            if verbose
            else range(len(self.trainer.test_dataset))
        )

        with torch.no_grad():
            for i in iterator:
                data = self.trainer.test_dataset[i]

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
                self.align_R0, self.align_s, self.align_t = self.trainer.get_pose_align()
                c2w_R = c2w[..., :3, :3] @ self.align_R0
                c2w_t = (c2w[..., :3, 3] - self.align_t) / self.align_s @ self.align_R0
                c2w_aligned = torch.cat([c2w_R, c2w_t[..., None]], dim=-1)
                image_id = data["image_id"]
                x = data["x"]
                y = data["y"]

                # Generate rays
                rays = generate_camera_rays(x, y, c2w_aligned, self.trainer.test_dataset)

                # Render image
                rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
                    self.trainer.radiance_field,
                    self.trainer.estimator,
                    rays,
                    # rendering options
                    near_plane=self.config.scene_config.near_plane,
                    render_step_size=self.trainer.render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=0.004,
                    alpha_thre=0.01,
                )

                # Compute metrics
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / torch.log(torch.tensor(10.0))
                lpips_score = self.trainer.lpips_fn(rgb, pixels)

                psnrs.append(psnr.item())
                lpips_scores.append(lpips_score.item())

                # Save images if requested
                if self.save_images and i == 0:
                    self._save_test_images(rgb, pixels, acc, depth, i)

        # Compute averages
        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips_scores) / len(lpips_scores)

        return {
            "psnr_avg": psnr_avg,
            "lpips_avg": lpips_avg,
            "psnrs": psnrs,
            "lpips_scores": lpips_scores,
        }