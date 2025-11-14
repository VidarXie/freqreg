"""
Few-shot camera subset utilities.

This module provides camera index selection strategies that aim to avoid
spatial clustering and encourage uniform coverage of the scene.
"""

from __future__ import annotations

import numpy as np
import torch


def camera_positions(c2w_np: np.ndarray) -> torch.Tensor:
    """Extract camera positions from c2w matrices.

    Args:
        c2w_np: [N, 3, 4] or [N, 4, 4] numpy array of camera-to-world matrices.

    Returns:
        torch.Tensor: [N, 3] camera centers (on CPU).
    """
    if c2w_np.shape[-2:] == (4, 4):
        pos = torch.from_numpy(c2w_np[:, :3, 3]).float()
    else:
        pos = torch.from_numpy(c2w_np[:, :3, 3]).float()
    return pos


def look_dirs(c2w_np: np.ndarray, opengl_camera: bool = True) -> torch.Tensor:
    """Extract forward/look direction vectors from c2w matrices.

    Uses the +Z axis for camera space; flip for OpenGL-style convention.

    Args:
        c2w_np: [N, 3, 4] or [N, 4, 4] numpy array.
        opengl_camera: if True, use -Z as the forward direction.

    Returns:
        torch.Tensor: [N, 3] unit look directions (not normalized here).
    """
    if c2w_np.shape[-2:] == (4, 4):
        z_col = torch.from_numpy(c2w_np[:, :3, 2]).float()
    else:
        z_col = torch.from_numpy(c2w_np[:, :3, 2]).float()
    return -z_col if opengl_camera else z_col


def _fps_indices(features: torch.Tensor, k: int, seed: int) -> np.ndarray:
    """Farthest Point Sampling on feature vectors [N, D] (CPU torch)."""
    N = features.shape[0]
    k = min(k, N)
    if k <= 0:
        return np.array([], dtype=np.int64)

    g = torch.Generator(device="cpu").manual_seed(seed)
    start = int(torch.randint(0, N, (1,), generator=g).item())

    selected = [start]
    # min-distance to the selected set
    dmin = torch.cdist(features[start : start + 1], features).squeeze(0)  # [N]
    for _ in range(1, k):
        nxt = int(torch.argmax(dmin).item())
        selected.append(nxt)
        dmin = torch.minimum(
            dmin, torch.cdist(features[nxt : nxt + 1], features).squeeze(0)
        )
    return np.array(selected, dtype=np.int64)


def choose_camera_indices(
    camtoworlds: np.ndarray,
    k: int,
    seed: int = 0,
) -> np.ndarray:
    """Select k camera indices with good coverage.

    modes:
        - "uniform_index"    : evenly spaced by index
        - "stratified_angle" : round-robin over azimuth bins
        - "fps_pose"         : farthest-point sampling on (position [+ lookdir])

    Args:
        camtoworlds: numpy array [N, 3, 4] or [N, 4, 4].
        k: number of cameras to keep.
        mode: selection strategy.
        angle_bins: for stratified_angle.
        orient_weight: >0 to include look directions (scaled) in FPS features.
        seed: RNG seed.

    Returns:
        np.ndarray of shape [k] with indices.
    """
    N = camtoworlds.shape[0]
    k = max(1, min(k, N))
    # fps_pose (default)
    pos = camera_positions(camtoworlds)  # [N,3] torch
    # Normalize positions for scale invariance
    pos = pos - pos.mean(0, keepdim=True)
    denom = pos.norm(dim=1, keepdim=True).median().clamp_min(1e-6)
    pos = pos / denom

    feats = pos  # [N,3]

    return _fps_indices(feats.contiguous(), k, seed)
