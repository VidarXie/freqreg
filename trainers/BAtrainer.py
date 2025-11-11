"""
NeRF trainer class for modular training pipeline.
"""

import time
from typing import Dict
import rerun as rr
from pathlib import Path
import math

import torch
import torch.nn.functional as F

from config import NeRFConfig
from utils.render_utils import (
    render_image_with_occgrid,
    generate_camera_rays_with_perturbation,
)

from trainers.trainer import NeRFTrainer, NGPRadianceField

from utils.lie_utils import LIE_
from utils.pose_utils import POSE_, sim3_align_errors
from utils.rerun import RerunLogger, create_blueprint
from utils.irls_utils import robust_weights_from_residuals

from nerfacc.estimators.occ_grid import OccGridEstimator


class BATrainer(NeRFTrainer):
    """
    Bundle Adjustment NeRF trainer with proposal networks.
    """

    def __init__(self, config: NeRFConfig):
        self.se3_noise_factor = 0.02
        super().__init__(config)
        self.start = 0.0
        self.end = 0.75
        self.init_level = 6.5
        if self.train_dataset.OPENGL_CAMERA:
            # for blender dataset
            self.init_level = 4.0

        self.rerun_logger = RerunLogger(Path("world"))
        blueprint = create_blueprint(Path("world"))
        rr.init("pose_refinement", spawn=True)
        rr.send_blueprint(blueprint)

        self.rerun_factor = 1.0
        if self.train_dataset.OPENGL_CAMERA:
            # for blender dataset
            self.rerun_factor = 8.0

        print("=" * 20)
        print("Using BA trainer")
        print("=" * 20)
        print("Noise factor:", self.se3_noise_factor)
        print("Initial level:", self.init_level)
        print("Rerun factor:", self.rerun_factor)
        print("=" * 20)

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
        self.se3_refine = torch.nn.Parameter(
            torch.zeros(len(self.train_dataset), 6, dtype=torch.float)
            .cuda()
            .requires_grad_(True)
        )
        # torch.nn.Embedding(len(self.train_dataset), 6).to(self.device)
        # torch.nn.init.zeros_(self.se3_refine.weight)

    def _setup_optimizers(self):
        """Initialize optimizers and schedulers."""
        self.optimizer = torch.optim.AdamW(
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

        self.pose_optimizer = torch.optim.AdamW([self.se3_refine], lr=0.001)
        if self.train_dataset.OPENGL_CAMERA:
            self.pose_optimizer = torch.optim.AdamW([self.se3_refine], lr=0.005)

        # Schedulers
        pose_opt_milestones = [
            self.config.training.max_steps // 2,
            self.config.training.max_steps * 3 // 4,
            self.config.training.max_steps * 5 // 6,
            self.config.training.max_steps * 9 // 10,
        ]

        self.pose_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.pose_optimizer,
                    start_factor=0.001,
                    total_iters=0.01 * self.config.training.max_steps,
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    self.pose_optimizer, milestones=pose_opt_milestones, gamma=0.33
                ),
            ]
        )

    def get_pose(self, c2w, noise, refine):
        # Add noise to camera pose
        c2w_pose = POSE_.from_matrix(c2w)  # [:,3,4]
        c2w_perturbed_pose = POSE_.compose_pair(c2w_pose, noise)
        c2w_refined_pose = POSE_.compose_pair(
            c2w_perturbed_pose, LIE_.se3_to_SE3(refine)
        )
        return c2w_refined_pose

    def get_pose_by_camera(self):
        c2w = self.train_dataset.camtoworlds
        return self.get_pose(c2w, self.se3_noise_pose, self.se3_refine)

    @torch.no_grad()
    def get_pose_error(self):
        est_poses = self.get_pose_by_camera()
        out_original, te_original, re_original, _, _, _ = sim3_align_errors(
            self.train_dataset.camtoworlds[..., :3, :4], est_poses
        )

        # Identify outlier poses
        q1 = torch.quantile(te_original, 0.25)
        q3 = torch.quantile(te_original, 0.75)
        iqr = q3 - q1
        upper_fence = q3 + 3.0 * iqr
        inlier_indices = (te_original < upper_fence).nonzero(as_tuple=True)[0]

        outlier_pct = 100.0 * (1.0 - len(inlier_indices) / len(te_original))

        out_inlier = torch.tensor(0.0)
        te_inlier = torch.tensor(0.0)
        re_inlier = torch.tensor(0.0)
        if len(inlier_indices) > 0:
            est_poses_in = est_poses[inlier_indices]
            gt_poses_in = self.train_dataset.camtoworlds[inlier_indices, :3, :4]
            out_inlier, te_inlier, re_inlier, _, _, _ = sim3_align_errors(
                gt_poses_in, est_poses_in
            )

        return (
            self.train_dataset.camtoworlds[..., :3, :4],
            out_original,
            te_original,
            re_original,
            out_inlier,
            te_inlier,
            re_inlier,
            outlier_pct,
        )

    @torch.no_grad()
    def get_pose_align(self):
        est_poses = self.get_pose_by_camera()
        out, te, re, _, _, _ = sim3_align_errors(
            self.train_dataset.camtoworlds[..., :3, :4], est_poses
        )

        # Identify outlier poses
        q1 = torch.quantile(te, 0.25)
        q3 = torch.quantile(te, 0.75)
        iqr = q3 - q1
        upper_fence = q3 + 3.0 * iqr
        inlier_indices = (te < upper_fence).nonzero(as_tuple=True)[0]

        est_poses_in = est_poses[inlier_indices]
        gt_poses_in = self.train_dataset.camtoworlds[inlier_indices, :3, :4]
        out_inlier, te_inlier, re_inlier, R0, s, t = sim3_align_errors(
            gt_poses_in, est_poses_in
        )

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
        t = min(progress / (self.end - self.start), 1.0)

        alpha = 0.5 * (1.0 + math.cos(2.0 * math.pi * (5.5 * t))) * math.exp(-3.0 * t)
        if self.train_dataset.OPENGL_CAMERA:
            alpha = (
                0.5 * (1.0 + math.cos(2.0 * math.pi * (3.5 * t))) * math.exp(-3.0 * t)
            )

        target_level = self.init_level * alpha

        # Sample training data
        i = torch.randint(0, len(self.train_dataset), (1,)).item()
        data = self.train_dataset[i]

        render_bkgd = data["color_bkgd"]
        pixels = data["pixels"]

        image_id = data["image_id"]
        x = data["x"]
        y = data["y"]

        c2w_refined_pose = self.get_pose_by_camera()[image_id]
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

        num_images = len(self.train_dataset)
        rays_per_image = len(pixels) // num_images
        rgb_reshaped = rgb.view(num_images, rays_per_image, -1)
        pixels_reshaped = pixels.view(num_images, rays_per_image, -1)
        mse_reshaped = F.mse_loss(rgb_reshaped, pixels_reshaped, reduction="none")
        mse_per_image = mse_reshaped.mean(dim=[1, 2])

        # Compute loss
        if alpha < 0.1:
            loss = self.mle_loss(rgb, pixels)
        elif alpha < 1.0 and alpha >= 0.1:
            loss = self.irls_loss(rgb, pixels, mse_per_image)
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

        log_mse = torch.log(mse_per_image + 1e-8)
        normalized_mse = torch.clamp(
            (log_mse - log_mse.min()) / (log_mse.max() - log_mse.min() + 1e-8),
            min=0.1,
        )

        sgld_noise = (
            torch.randn_like(self.se3_refine)
            * normalized_mse[:, None]
            * self.pose_optimizer.param_groups[0]["lr"]
        )

        warmup_steps = int(0.06 * self.config.training.max_steps)

        self.se3_refine.data += sgld_noise * min(1.0, self.step / warmup_steps)

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

    def irls_loss(self, rgb, pixels, mse_per_image):
        eps = 1e-8
        rgb_render = torch.clamp(rgb, min=eps)
        rgb_gt = torch.clamp(pixels, min=eps)

        # --- IRLS weights from a image wise residual ---
        w_image, _ = robust_weights_from_residuals(mse_per_image)  # [N] in [0,1]
        w_image = w_image.clamp_min(0.0)

        num_images = len(self.train_dataset)
        rays_per_image = len(pixels) // num_images

        w_ray = w_image.repeat_interleave(
            rays_per_image
        )  # [num_images * rays_per_image]
        W = w_ray[:, None].expand_as(rgb_render)

        def wmean(x):
            return (W * x).sum() / (W.sum() + eps)

        # Data term:  E_w[ -log f * y ] / E_w[ y ]
        data_term = wmean(-torch.log(rgb_render) * rgb_gt) / (wmean(rgb_gt) + eps)

        # Normalizer term: log E_w[f]
        norm_term = torch.log(wmean(rgb_render) + eps)

        # Mean-matching (scale fix): (E_w[f] - E_w[y])^2
        mean_match = (wmean(rgb_render) - wmean(rgb_gt)) ** 2

        return data_term + norm_term + mean_match

    def print_training_stats(self, metrics: Dict[str, float]):
        """Print training statistics."""
        elapsed_time = time.time() - self.start_time

        gt_poses, est, te, re, est_inlier, te_inlier, re_inlier, outlier_pct = (
            self.get_pose_error()
        )

        print(
            f"elapsed_time={elapsed_time:.2f}s | step={self.step} | "
            f"loss={metrics['loss']:.5f} | psnr={metrics['psnr']:.2f} | "
            f"num_rays={metrics['num_rays']:d} | "
            f"translation_error={te.mean().item():.6f} | "
            f"rotation_error={re.mean().item():.6f} | "
            f"translation_error_inlier={te_inlier.mean().item():.6f} | "
            f"rotation_error_inlier={re_inlier.mean().item():.6f} | "
            f"outlier_pct={outlier_pct:.6f}"
        )

        if self.train_dataset.OPENGL_CAMERA:
            gt_poses = gt_poses.clone()
            est = est.clone()
            gt_poses[..., :3, 1:3] *= -1.0
            est[..., :3, 1:3] *= -1.0

        self.rerun_logger.log_poses_at_frame(
            gt_poses, est, self.step, self.rerun_factor
        )
