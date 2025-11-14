from __future__ import annotations

import torch

# Robust imports whether this file lives in a package or next to nerf_synthetic.py

from datasets.nerf_360_v2 import SubjectLoader as _BaseSubjectLoader
from datasets.fewshot_utils import choose_camera_indices


class FewShotColmapLoader(_BaseSubjectLoader):
    """Few-shot variant of the COLMAP SubjectLoader.

    Extra args:
        n_cams: If provided and < number of available images in the split,
                keep only n_cams images (chosen to be well-distributed).
        seed: RNG seed for deterministic starting point in FPS.
        apply_to_eval: If True, apply subsampling also on the 'test' split.
                       By default, we only subsample during training.
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
        factor: int = 1,
        device: str = "cpu",
        # ---- Few-shot extras ----
        n_cams: int = 50,
        seed: int = 0,
        apply_to_eval: bool = False,
    ) -> None:
        # 1) Build the full dataset first (loads, normalizes, splits, tensors ready).
        super().__init__(
            subject_id=subject_id,
            root_fp=root_fp,
            split=split,
            color_bkgd_aug=color_bkgd_aug,
            num_rays=num_rays,
            near=near,
            far=far,
            factor=factor,
            device=device,
        )

        # 2) Do nothing if no subsampling requested / not beneficial.
        if n_cams is None or n_cams >= len(self.images):
            return

        # 3) Apply subsampling for training by default (or always if apply_to_eval=True).
        if not (self.training or apply_to_eval):
            return

        # 4) Build feature space using camtoworlds (now a torch tensor on device).
        #    For COLMAP/OpenCV convention we use +Z as forward, so default orient_weight=0.
        c2w_np = self.camtoworlds.detach().cpu().numpy()  # [N, 3, 4] or [N, 4, 4]
        idx_np = choose_camera_indices(
            camtoworlds=c2w_np,
            k=int(n_cams),
            seed=int(seed),
        )

        # 5) Index-select on GPU/CPU tensors.
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=self.images.device)
        self.images = self.images.index_select(0, idx)
        self.camtoworlds = self.camtoworlds.index_select(0, idx)

        # 6) Update per-epoch bookkeeping for ray sampling.
        if hasattr(self, "num_rays"):
            if self.num_rays:
                self.rays_per_image = self.num_rays // len(self.images)
                if self.rays_per_image < 1:
                    self.rays_per_image = 1
                self.total_rays = self.rays_per_image * len(self.images)
            else:
                self.rays_per_image = None
                self.total_rays = None
