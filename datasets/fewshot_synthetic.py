"""
Few-shot NeRF-Synthetic loader (subclass of the existing SubjectLoader).

- Adds uniform camera subsampling to enable few-shot training.
- Keeps the rest of the pipeline intact (ray sampling, batching, etc.).

Usage:
    from fewshot_synthetic import FewShotSyntheticLoader as SubjectLoader
    # or import alongside your original loader and switch per experiment

    ds = FewShotSyntheticLoader(
        subject_id="lego",
        root_fp="./data/nerf_synthetic",
        split="train",
        num_rays=8192,          # same as original
        n_cams=6,               # enable few-shot (pick 6 cameras)
        cam_sampler="fps_pose", # "uniform_index" | "stratified_angle" | "fps_pose"
        angle_bins=12,
        orient_weight=0.3,
        seed=123,
        device=torch.device("cuda"),
    )
"""

from __future__ import annotations

from typing import Optional, Literal

import numpy as np
import torch

# Robust imports whether this file lives in a package or next to nerf_synthetic.py

from datasets.nerf_synthetic import SubjectLoader as _BaseSubjectLoader
from datasets.fewshot_utils import choose_camera_indices


class FewShotSyntheticLoader(_BaseSubjectLoader):
    """A drop-in few-shot variant of the original SubjectLoader.

    Extra Args (beyond the base class):
        n_cams: if provided and < #images, keep only n_cams images.
        cam_sampler: camera subset strategy:
            - "uniform_index": evenly spaced by frame index
            - "stratified_angle": round-robin over azimuth bins
            - "fps_pose": farthest point sampling on pose features (default)
        angle_bins: number of bins for "stratified_angle".
        orient_weight: weight for look direction in "fps_pose" features (0 = positions only).
        seed: RNG seed for deterministic selection.
        apply_to_eval: if True, also apply subsampling on eval/inference splits.
    """

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int | None = None,
        near: float | None = None,
        far: float | None = None,
        device: torch.device = torch.device("cpu"),
        # ---- Few-shot extras ----
        n_cams: int = 20,
        cam_sampler: Literal[
            "uniform_index", "stratified_angle", "fps_pose"
        ] = "fps_pose",
        angle_bins: int = 8,
        orient_weight: float = 0.0,
        seed: int = 0,
        apply_to_eval: bool = False,
    ) -> None:
        # Initialize the base loader (loads all images first).
        super().__init__(
            subject_id=subject_id,
            root_fp=root_fp,
            split=split,
            color_bkgd_aug=color_bkgd_aug,
            num_rays=num_rays,
            near=near,
            far=far,
            device=device,
        )

        # Only apply subsampling if requested and beneficial.
        if n_cams is None or n_cams >= len(self.images):
            return

        # By default, apply during training only (unless apply_to_eval=True).
        if not (self.training or apply_to_eval):
            return

        # Select indices using pose information from self.camtoworlds.
        # self.camtoworlds is a torch tensor [N, 3, 4] on device.
        c2w_np = self.camtoworlds.detach().cpu().numpy()
        idx_np = choose_camera_indices(
            camtoworlds=c2w_np,
            k=int(n_cams),
            mode=cam_sampler,
            angle_bins=int(angle_bins),
            orient_weight=float(orient_weight),
            seed=int(seed),
        )

        # Convert indices to a device tensor for fast gather
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=self.images.device)

        # Subset images and poses
        self.images = self.images.index_select(0, idx)
        self.camtoworlds = self.camtoworlds.index_select(0, idx)

        # NOTE: Intrinsics (K, focal, width, height) remain the same.
        # Recompute per-epoch ray bookkeeping with the new number of images.
        # These attributes exist in the base class; recompute them consistently.
        if hasattr(self, "num_rays"):
            if self.num_rays:
                self.rays_per_image = self.num_rays // len(self.images)
                # Ensure at least 1 ray per image if num_rays is tiny
                if self.rays_per_image < 1:
                    self.rays_per_image = 1
                self.total_rays = self.rays_per_image * len(self.images)
            else:
                self.rays_per_image = None
                self.total_rays = None
