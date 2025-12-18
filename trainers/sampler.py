import os
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import tqdm
from trainers.trainer import NeRFTrainer
from utils.render_utils import (
    generate_camera_rays_with_perturbation_sampling,
    render_image_with_occgrid,
)

from utils.lie_utils import LIE_
from utils.pose_utils import POSE_


class NeRFSampler:
    def __init__(self, trainer: NeRFTrainer):
        self.se3_noise_factor = 0.3
        self.trainer = trainer
        self.config = trainer.config
        self.max_steps = 100
        self._setup_pertubation()

        self.rate_decay = 0.95

        self.device = self.trainer.device

        self.base_pose = None

        self.output_dir = self.config.output_dir

    def _setup_pertubation(self):
        se3_noise = (
            torch.randn(len(self.trainer.test_dataset), 6, device=self.config.device)
            * self.se3_noise_factor
        )
        self.se3_noise_pose = LIE_.se3_to_SE3(se3_noise)

    def get_pose(self, c2w, noise, refine):
        # Add noise to camera pose
        c2w_pose = POSE_.from_matrix(c2w)  # [:,3,4]
        c2w_perturbed_pose = POSE_.compose_pair(c2w_pose, noise)
        c2w_refined_pose = POSE_.compose_pair(
            c2w_perturbed_pose, LIE_.se3_to_SE3(refine)
        )
        return c2w_refined_pose

    def _pose_from_twist(self, xi):
        # twist -> SE3 delta
        T_delta = LIE_.se3_to_SE3(xi[None, :])  # [1,4,4]
        pose_i = POSE_.compose_pair(self.base_pose, T_delta)  # POSE_ object
        c2w_i = POSE_.to_matrix(pose_i)  # [1,4,4]
        return c2w_i

    def _energy_likelihood(self, xi, data, current_level):
        sample_lvl = 4
        render_bkgd = data["color_bkgd"]
        pixels = data["pixels"][::sample_lvl, ::sample_lvl, :]

        x = data["x"].view(
            self.trainer.test_dataset.height, self.trainer.test_dataset.width
        )
        x = x[::sample_lvl, ::sample_lvl].flatten()
        y = data["y"].view(
            self.trainer.test_dataset.height, self.trainer.test_dataset.width
        )
        y = y[::sample_lvl, ::sample_lvl].flatten()

        c2w_i = self._pose_from_twist(xi)

        with torch.no_grad():
            # Generate rays
            rays = generate_camera_rays_with_perturbation_sampling(
                x,
                y,
                c2w_i,
                self.trainer.test_dataset,
                mip_level=current_level,
                sampling_level=sample_lvl,
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
            eps = 1e-8
            rgb_render = torch.clamp(rgb, min=eps)
            rgb_gt = torch.clamp(pixels, min=eps)

            energy = torch.mean(-torch.log(rgb_render) * rgb_gt) / torch.mean(rgb_gt)
            energy += torch.log(torch.mean(rgb_render))
            energy += (torch.mean(rgb_render) - torch.mean(rgb_gt)) ** 2

            mse_loss = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse_loss) / torch.log(torch.tensor(10.0))

        return mse_loss.item(), psnr.item()  # Python float

    def sampling(
        self,
        index: int,
        num_iters: int = 100,
        num_samples: int = 256,
        num_elites: int = 32,
        init_trans_sigma: float = 0.3,
        init_rot_sigma: float = 0.3,
        cov_shrink: float = 0.95,
    ):
        self.trainer.radiance_field.eval()
        self.trainer.estimator.eval()

        data = self.trainer.test_dataset[index]

        c2w = data["c2w"]

        pose_perturbation = self.se3_noise_pose[index]

        self.base_pose = POSE_.compose_pair(POSE_.from_matrix(c2w), pose_perturbation)

        t_err, r_err = self._transformation_error(
            POSE_.to_matrix(self.base_pose), c2w[None, :, :]
        )

        print(
            f"Initial Translation error: {t_err.item():.4f}, Initial Rotation error: {r_err.item():.4f} rad"
        )

        # Initial Gaussian in twist space
        dim = 6
        mu = torch.zeros(dim, device=self.device)
        Sigma = torch.zeros(dim, dim, device=self.device)

        # translation variance
        Sigma[0:3, 0:3] = (init_trans_sigma**2) * torch.eye(3, device=self.device)
        # rotation variance
        Sigma[3:6, 3:6] = (init_rot_sigma**2) * torch.eye(3, device=self.device)

        best_xi = None
        best_E = float("inf")
        best_psnr = None
        current_level = 4.0

        t_err_history = []
        r_err_history = []
        # Early stop setting
        best_total_err = float("inf")
        last_improve_iter = 0
        patience = 15  # stop if no improvement for 10 iterations
        min_improve = 1e-4  # required improvement to count as "better"

        for it in range(num_iters):
            # Cholesky for sampling
            L = torch.linalg.cholesky(Sigma + 1e-6 * torch.eye(dim, device=self.device))

            xis = []
            energies = []
            psnrs = []
            for n in tqdm.tqdm(
                range(num_samples), desc=f"CEM iter {it + 1}", leave=False
            ):
                z = torch.randn(dim, device=self.device)
                xi = mu + L @ z
                xis.append(xi)

                E, psnr = self._energy_likelihood(xi, data, current_level)
                energies.append(E)
                psnrs.append(psnr)

                if E < best_E:
                    best_E = E
                    best_xi = xi.clone()
                    best_psnr = psnr

            energies = torch.tensor(energies, device=self.device)
            xis = torch.stack(xis, dim=0)  # [N,6]

            # Select elites
            elite_idx = torch.topk(-energies, k=num_elites).indices  # smallest E
            elites = xis[elite_idx]  # [M,6]

            # Refit Gaussian
            new_mu = elites.mean(dim=0)
            diff = elites - new_mu
            new_Sigma = (diff[:, :, None] * diff[:, None, :]).mean(dim=0)

            # Shrink covariance to stabilize
            Sigma = cov_shrink * new_Sigma + (1.0 - cov_shrink) * Sigma
            mu = new_mu

            # Optional: anneal mip level (coarse → fine)
            current_level = max(0.0, current_level * 0.9)

            # Measure the difference between current pose and original pose
            t_err, r_err = self._transformation_error(
                self._pose_from_twist(best_xi), c2w[None, :, :]
            )

            print(
                f"Translation error: {t_err.item():.4f}, Rotation error: {r_err.item():.4f} rad, PSNR: {best_psnr:.4f}"
            )

            t_err_history.append(t_err.item())
            r_err_history.append(r_err.item())

            total_err = t_err.item() + r_err.item()
            if total_err < best_total_err - min_improve:
                best_total_err = total_err
                last_improve_iter = it
            else:
                if it - last_improve_iter >= patience:
                    print(
                        f"[Early stop] No improvement for {patience} iterations. "
                        f"Best total error: {best_total_err:.6f}"
                    )
                    break

        output_dir = self.output_dir + "_sampling_wo"
        test_images_dir = os.path.join(output_dir, f"{index}")
        self._save_lists(t_err_history, r_err_history, test_images_dir)

    def _transformation_error(self, A, B):
        assert A.shape[-2:] == (4, 4) and B.shape[-2:] == (4, 4), (
            f"Expected [..., 4, 4], got {A.shape} and {B.shape}"
        )

        eps = 1e-7

        # ---- Translation error ----
        t_A = A[..., :3, 3]
        t_B = B[..., :3, 3]
        d_t = t_A - t_B
        trans_err = torch.linalg.norm(
            d_t, dim=-1
        )  # [...], in meters (or whatever units)

        # ---- Rotation error ----
        R_A = A[..., :3, :3]
        R_B = B[..., :3, :3]

        # relative rotation: R_rel = R_A^T * R_B
        R_rel = R_A.transpose(-2, -1) @ R_B

        # trace-based angle: cos θ = (tr(R_rel) - 1) / 2
        trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
        cos_theta = (trace - 1.0) * 0.5
        cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
        rot_err = torch.acos(cos_theta)  # [...], radians

        return trans_err, rot_err

    def _save_lists(self, list_a, list_b, outpath_folder):
        out_dir = Path(outpath_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Human-readable: JSON
        json_path = out_dir / "lists.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"list_a": list_a, "list_b": list_b}, f, indent=2)
