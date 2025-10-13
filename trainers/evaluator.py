"""
NeRF evaluator class for model evaluation and metrics computation.
"""

import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
from typing import Dict
import torch
import torch.nn.functional as F
import tqdm
from trainers.trainer import NeRFTrainer
from utils.render_utils import (
    render_image_with_propnet,
    generate_camera_rays,
    render_image_with_occgrid,
)


class NeRFEvaluator:
    """
    Handles evaluation of trained NeRF models.
    """

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
            self.test_images_dir = os.path.join(self.config.output_dir, "test_images")
            os.makedirs(self.test_images_dir, exist_ok=True)

    def evaluate(self, verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate the model on test dataset.

        Args:
            verbose: Whether to show progress bar

        Returns:
            Dictionary containing average PSNR and LPIPS scores
        """
        # Set models to evaluation mode
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
                x = data["x"]
                y = data["y"]

                # Generate rays
                rays = generate_camera_rays(x, y, c2w, self.trainer.test_dataset)

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

    def _save_test_images(
        self,
        rgb: torch.Tensor,
        pixels: torch.Tensor,
        acc: torch.Tensor,
        depth: torch.Tensor,
        image_idx: int,
    ):
        """Save rendered test images and error maps."""

        # Convert to numpy
        rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
        pixels_np = (pixels.cpu().numpy() * 255).astype(np.uint8)

        # Handle depth tensor - it might have shape [H, W, 1] or [H, W]
        if depth.dim() == 3 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)

        # Compute error map
        error = (rgb - pixels).norm(dim=-1).cpu().numpy()
        error_np = (error * 255).astype(np.uint8)

        # Convert single-channel images to 3-channel for concatenation
        error_np_3ch = np.stack([error_np] * 3, axis=-1)

        # Convert depth to numpy (normalize to [0, 255])
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
        depth_np = (depth_normalized.cpu().numpy() * 255).astype(np.uint8)
        cmap = (
            plt.cm.viridis
        )  # You can change to plt.cm.jet or other colormaps like plt.cm.plasma
        depth_colored = cmap(depth_np / 255.0)
        depth_np_3ch = (depth_colored[:, :, :3] * 255).astype(np.uint8)

        # Concatenate images horizontally: rgb, gt, error, acc, depth
        combined = np.concatenate(
            [pixels_np, rgb_np, error_np_3ch, depth_np_3ch], axis=1
        )

        # Save combined image
        imageio.imwrite(
            os.path.join(
                self.test_images_dir,
                f"test_combined_{image_idx:03d}_{self.trainer.step}.png",
            ),
            combined,
        )

    def evaluate_single_image(self, image_idx: int) -> Dict[str, float]:
        """
        Evaluate a single image from test dataset.

        Args:
            image_idx: Index of image to evaluate

        Returns:
            Dictionary containing metrics for this image
        """
        if image_idx >= len(self.trainer.test_dataset):
            raise ValueError(f"Image index {image_idx} out of range")

        # Set models to evaluation mode
        self.trainer.radiance_field.eval()
        for p in self.trainer.proposal_networks:
            p.eval()
        self.trainer.estimator.eval()

        with torch.no_grad():
            data = self.trainer.test_dataset[image_idx]
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            # Render image
            rgb, acc, depth, extras = render_image_with_propnet(
                self.trainer.radiance_field,
                self.trainer.proposal_networks,
                self.trainer.estimator,
                rays,
                # rendering options
                num_samples=self.config.model.num_samples,
                num_samples_per_prop=self.config.model.num_samples_per_prop,
                near_plane=self.config.scene_config.near_plane,
                far_plane=self.config.scene_config.far_plane,
                sampling_type=self.config.model.sampling_type,
                opaque_bkgd=self.config.model.opaque_bkgd,
                render_bkgd=render_bkgd,
                # test options
                test_chunk_size=self.config.test_chunk_size,
            )

            # Compute metrics
            mse = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse) / torch.log(torch.tensor(10.0))
            lpips_score = self.trainer.lpips_fn(rgb, pixels)

            return {
                "psnr": psnr.item(),
                "lpips": lpips_score.item(),
                "mse": mse.item(),
                "rgb": rgb,
                "acc": acc,
                "depth": depth,
                "pixels": pixels,
                "extras": extras,
            }

    def print_evaluation_results(self, results: Dict[str, float]):
        """Print evaluation results in a formatted way."""
        print("Evaluation Results:")
        print(f"  Average PSNR: {results['psnr_avg']:.3f}")
        print(f"  Average LPIPS: {results['lpips_avg']:.4f}")
        if "psnrs" in results:
            psnrs = results["psnrs"]
            print(f"  PSNR std: {torch.tensor(psnrs).std().item():.3f}")
        if "lpips_scores" in results:
            lpips_scores = results["lpips_scores"]
            print(f"  LPIPS std: {torch.tensor(lpips_scores).std().item():.4f}")
