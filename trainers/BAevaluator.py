"""
NeRF evaluator class for modular training pipeline.
"""

import tqdm

import torch
import torch.nn.functional as F

from utils.render_utils import (
    render_image_with_occgrid,
    generate_camera_rays,
)

from trainers.evaluator import NeRFEvaluator


class BAEvaluator(NeRFEvaluator):
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
                c2w_R = torch.einsum("ij,njk->nik", align_R0.t(), c2w[..., :3, :3])
                c2w_t = (c2w[..., :3, 3] - align_t) / align_s @ align_R0
                c2w_aligned = torch.cat([c2w_R, c2w_t[..., None]], dim=-1)
                image_id = data["image_id"]
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

        # Compute averages
        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips_scores) / len(lpips_scores)

        return {
            "psnr_avg": psnr_avg,
            "lpips_avg": lpips_avg,
            "psnrs": psnrs,
            "lpips_scores": lpips_scores,
        }
