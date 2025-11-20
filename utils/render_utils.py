"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import random
from typing import Optional, Sequence

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
from datasets.utils import Rays, namedtuple_map
from torch.utils.data._utils.collate import collate, default_collate_fn_map
import torch.nn.functional as F

from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.estimators.prop_net import PropNetEstimator
from nerfacc.volrend import (
    rendering,
)

NERF_SYNTHETIC_SCENES = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
    "start"
]
MIPNERF360_UNBOUNDED_SCENES = [
    "garden",
    "bicycle",
    "bonsai",
    "counter",
    "kitchen",
    "room",
    "stump",
    "flowers",
    "treehill",
    "output"
]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render_image_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)
    else:
        num_rays, _ = rays_shape

    results = []
    chunk = torch.iinfo(torch.int32).max if radiance_field.training else test_chunk_size
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)

        rays_o = chunk_rays.origins
        rays_d = chunk_rays.viewdirs

        def sigma_fn(t_starts, t_ends, ray_indices):
            if t_starts.shape[0] == 0:
                sigmas = torch.empty((0, 1), device=t_starts.device)
            else:
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                if timestamps is not None:
                    # dnerf
                    t = (
                        timestamps[ray_indices]
                        if radiance_field.training
                        else timestamps.expand_as(positions[:, :1])
                    )
                    sigmas = radiance_field.query_density(positions, t)
                else:
                    sigmas = radiance_field.query_density(positions)
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            if t_starts.shape[0] == 0:
                rgbs = torch.empty((0, 3), device=t_starts.device)
                sigmas = torch.empty((0, 1), device=t_starts.device)
            else:
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                if timestamps is not None:
                    # dnerf
                    t = (
                        timestamps[ray_indices]
                        if radiance_field.training
                        else timestamps.expand_as(positions[:, :1])
                    )
                    rgbs, sigmas = radiance_field(positions, t, t_dirs)
                else:
                    rgbs, sigmas = radiance_field(positions, t_dirs)
            return rgbs, sigmas.squeeze(-1)

        ray_indices, t_starts, t_ends = estimator.sampling(
            rays_o.detach(),
            rays_d.detach(),
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=rays_o.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )


def render_image_with_propnet(
    # scene
    radiance_field: torch.nn.Module,
    proposal_networks: Sequence[torch.nn.Module],
    estimator: PropNetEstimator,
    rays: Rays,
    # rendering options
    num_samples: int,
    num_samples_per_prop: Sequence[int],
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    sampling_type: Literal["uniform", "lindisp"] = "lindisp",
    opaque_bkgd: bool = True,
    render_bkgd: Optional[torch.Tensor] = None,
    # train options
    proposal_requires_grad: bool = False,
    # test options
    test_chunk_size: int = 8192,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)
    else:
        num_rays, _ = rays_shape

    def prop_sigma_fn(t_starts, t_ends, proposal_network):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :]
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        sigmas = proposal_network(positions)
        if opaque_bkgd:
            sigmas[..., -1, :] = torch.inf
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :].repeat_interleave(
            t_starts.shape[-1], dim=-2
        )
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        rgb, sigmas = radiance_field(positions, t_dirs)
        if opaque_bkgd:
            sigmas[..., -1, :] = torch.inf
        return rgb, sigmas.squeeze(-1)

    results = []
    chunk = torch.iinfo(torch.int32).max if radiance_field.training else test_chunk_size
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        t_starts, t_ends = estimator.sampling(
            prop_sigma_fns=[
                lambda *args: prop_sigma_fn(*args, p) for p in proposal_networks
            ],
            prop_samples=num_samples_per_prop,
            num_samples=num_samples,
            n_rays=chunk_rays.origins.shape[0],
            near_plane=near_plane,
            far_plane=far_plane,
            sampling_type=sampling_type,
            stratified=radiance_field.training,
            requires_grad=False,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices=None,
            n_rays=None,
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth]
        results.append(chunk_results)

    colors, opacities, depths = collate(
        results,
        collate_fn_map={
            **default_collate_fn_map,
            torch.Tensor: lambda x, **_: torch.cat(x, 0),
        },
    )
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        extras,
    )


def generate_camera_rays(x, y, c2w, subject) -> Rays:
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - subject.K[0, 2] + 0.5) / subject.K[0, 0],
                (y - subject.K[1, 2] + 0.5)
                / subject.K[1, 1]
                * (-1.0 if subject.OPENGL_CAMERA else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if subject.OPENGL_CAMERA else 1.0),
    )  # [num_rays, 3]

    # [n_cams, height, width, 3]
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)

    if subject.training:
        origins = torch.reshape(origins, (subject.total_rays, 3))
        viewdirs = torch.reshape(viewdirs, (subject.total_rays, 3))
    else:
        origins = torch.reshape(origins, (subject.height, subject.width, 3))
        viewdirs = torch.reshape(viewdirs, (subject.height, subject.width, 3))

    rays = Rays(origins=origins, viewdirs=viewdirs)
    return rays


def generate_camera_rays_with_perturbation(
    x, y, c2w, subject, mip_level: int = 0
) -> Rays:
    """
    Generate camera rays with pixel coordinate perturbation for mip-mapping.

    Args:
        x: X pixel coordinates tensor
        y: Y pixel coordinates tensor
        c2w: Camera-to-world transformation matrix
        subject: Dataset object containing camera intrinsics and image dimensions
        mip_level: Mipmap level for perturbation (0 = no perturbation, >0 = increasing perturbation)

    Returns:
        Rays object containing origins and view directions with perturbation applied
    """
    # Calculate perturbation std based on mipmap level
    # For mip_level <= 0, no perturbation (perturbation_std = 0)
    # For mip_level > 0, perturbation scales with 2^mip_level
    if mip_level <= 0:
        perturbation_std = 0.0
    else:
        perturbation_std = 2.0**mip_level

    # Apply perturbation to pixel coordinates
    if perturbation_std > 0:
        x_noise = torch.randn_like(x.float()) * perturbation_std
        y_noise = torch.randn_like(y.float()) * perturbation_std
        x_perturbed = x.float() + x_noise
        y_perturbed = y.float() + y_noise
    else:
        x_perturbed = x.float()
        y_perturbed = y.float()

    # Generate camera directions using perturbed coordinates
    camera_dirs = F.pad(
        torch.stack(
            [
                (x_perturbed - subject.K[0, 2] + 0.5) / subject.K[0, 0],
                (y_perturbed - subject.K[1, 2] + 0.5)
                / subject.K[1, 1]
                * (-1.0 if subject.OPENGL_CAMERA else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if subject.OPENGL_CAMERA else 1.0),
    )  # [num_rays, 3]

    # [n_cams, height, width, 3]
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)

    if subject.training:
        origins = torch.reshape(origins, (subject.total_rays, 3))
        viewdirs = torch.reshape(viewdirs, (subject.total_rays, 3))
    else:
        origins = torch.reshape(origins, (subject.height, subject.width, 3))
        viewdirs = torch.reshape(viewdirs, (subject.height, subject.width, 3))

    rays = Rays(origins=origins, viewdirs=viewdirs)
    return rays
