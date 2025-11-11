"""
Few-shot camera subset utilities.

This module provides camera index selection strategies that aim to avoid
spatial clustering and encourage uniform coverage of the scene.
"""

from __future__ import annotations

from typing import Literal, Optional
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


def uniform_index_indices(n: int, k: int) -> np.ndarray:
    """Evenly spaced indices from [0, n-1]."""
    if k <= 1:
        return np.array([0], dtype=np.int64)
    idx = np.round(np.linspace(0, n - 1, num=k)).astype(np.int64)
    return np.unique(idx)


def stratified_angle_indices(
    c2w_np: np.ndarray, k: int, bins: int, seed: int
) -> np.ndarray:
    """Round-robin selection over azimuth angle bins to prevent clustering.

    Args:
        c2w_np: [N, 3, 4] camera-to-world matrices.
        k: number of cameras to select.
        bins: number of azimuth bins.
        seed: RNG seed for tie-breaking inside bins.

    Returns:
        np.ndarray of shape [k] with selected indices.
    """
    g = np.random.default_rng(seed)
    pos = camera_positions(c2w_np).numpy()  # [N,3]
    # azimuth around the origin (x,z)
    ang = (np.arctan2(pos[:, 0], pos[:, 2]) + 2 * np.pi) % (2 * np.pi)  # [0, 2π)
    bin_ids = np.floor(bins * ang / (2 * np.pi)).astype(int)
    N = len(pos)
    per_bin = [[] for _ in range(bins)]
    for i in range(N):
        per_bin[bin_ids[i]].append(i)

    # shuffle each bin so we don't always start with the first frame
    for b in range(bins):
        g.shuffle(per_bin[b])

    chosen = []
    ptr = [0] * bins
    while len(chosen) < min(k, N) and any(
        ptr[b] < len(per_bin[b]) for b in range(bins)
    ):
        for b in range(bins):
            if ptr[b] < len(per_bin[b]) and len(chosen) < k:
                chosen.append(per_bin[b][ptr[b]])
                ptr[b] += 1
    return np.array(chosen, dtype=np.int64)


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
    mode: Literal["uniform_index", "stratified_angle", "fps_pose"] = "fps_pose",
    angle_bins: int = 8,
    orient_weight: float = 0.0,
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

    if mode == "uniform_index":
        return uniform_index_indices(N, k)

    if mode == "stratified_angle":
        return stratified_angle_indices(camtoworlds, k, angle_bins, seed)

    # fps_pose (default)
    pos = camera_positions(camtoworlds)  # [N,3] torch
    # Normalize positions for scale invariance
    pos = pos - pos.mean(0, keepdim=True)
    denom = pos.norm(dim=1, keepdim=True).median().clamp_min(1e-6)
    pos = pos / denom

    if orient_weight > 0.0:
        look = look_dirs(camtoworlds, opengl_camera=True)  # [N,3]
        look = look / (look.norm(dim=1, keepdim=True).clamp_min(1e-6))
        feats = torch.cat([pos, orient_weight * look], dim=1)  # [N,6]
    else:
        feats = pos  # [N,3]

    return _fps_indices(feats.contiguous(), k, seed)
