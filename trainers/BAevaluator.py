"""
NeRF evaluator class for modular training pipeline.
"""

import tqdm
import os
import torch
import torch.nn.functional as F

from utils.render_utils import (
    render_image_with_occgrid,
    generate_camera_rays,
)

from trainers.evaluator import NeRFEvaluator
from trainers.trainer import NeRFTrainer


class BAEvaluator(NeRFEvaluator):
    def __init__(self, trainer: NeRFTrainer, save_images: bool = True):
        """
        Initialize evaluator.

        Args:
            trainer: NeRFTrainer instance containing models and config
            save_images: Whether to save rendered images during evaluation
        """
        self.trainer = trainer
        self.config = trainer.config
        self.save_images = save_images

        if self.save_images:
            output_dir = (
                self.config.output_dir
                + "_noise_"
                + f"{int(self.trainer.se3_noise_factor * 100):03d}"
            )
            self.test_images_dir = os.path.join(output_dir, "test_images")
            os.makedirs(self.test_images_dir, exist_ok=True)

    def _save_pose_errors(self):
        gt_poses, est, te, re, est_inlier, te_inlier, re_inlier, outlier_pct = (
            self.trainer.get_pose_error()
        )

        with open(os.path.join(self.test_images_dir, "pose_errors.txt"), "a") as f:
            f.write("=" * 50 + "\n")
            f.write(f"Step {self.trainer.step}: \n")
            f.write(
                f"translation_error={te.mean().item():.6f} | "
                f"rotation_error={re.mean().item():.6f} | "
                f"translation_error_inlier={te_inlier.mean().item():.6f} | "
                f"rotation_error_inlier={re_inlier.mean().item():.6f} | "
                f"outlier_pct={outlier_pct:.6f}"
            )
            f.write("\n")

    def evaluate(self, verbose=True):
        self.trainer.radiance_field.eval()
        self.trainer.estimator.eval()

        psnrs = []
        lpips_scores = []

        iterator = (
            tqdm.tqdm(range(len(self.trainer.test_dataset)), desc="Evaluating")
            if verbose
            else range(len(self.trainer.test_dataset))
        )
        align_R0, align_s, align_t = self.trainer.get_pose_align()
        with torch.no_grad():
            for i in iterator:
                data = self.trainer.test_dataset[i]

                render_bkgd = data["color_bkgd"]
                pixels = data["pixels"]

                c2w = data["c2w"]
                c2w_R = torch.einsum("ij,njk->nik", align_R0.t(), c2w[..., :3, :3])
                c2w_t = (c2w[..., :3, 3] - align_t) / align_s @ align_R0
                c2w_aligned = torch.cat([c2w_R, c2w_t[..., None]], dim=-1)
                x = data["x"]
                y = data["y"]

                # Generate rays
                rays = generate_camera_rays(
                    x, y, c2w_aligned, self.trainer.test_dataset
                )

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
                    self._save_pose_errors()

        # Compute averages
        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips_scores) / len(lpips_scores)

        return {
            "psnr_avg": psnr_avg,
            "lpips_avg": lpips_avg,
            "psnrs": psnrs,
            "lpips_scores": lpips_scores,
        }
