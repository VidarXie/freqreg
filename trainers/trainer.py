"""
NeRF trainer class for modular training pipeline.
"""

import time
from typing import Dict

import torch
import torch.nn.functional as F
from lpips import LPIPS

from config import NeRFConfig
from radiance_fields.ngp import NGPDensityField, NGPRadianceField
from utils.render_utils import (
    render_image_with_propnet,
    set_random_seed,
    generate_camera_rays,
)
from nerfacc.estimators.prop_net import (
    PropNetEstimator,
    get_proposal_requires_grad_fn,
)


class NeRFTrainer:
    """
    Modular NeRF trainer with proposal networks.
    """

    def __init__(self, config: NeRFConfig):
        """
        Initialize the NeRF trainer.

        Args:
            config: NeRFConfig containing all training parameters
        """
        self.config = config
        self.device = config.device

        # Set random seed
        set_random_seed(config.seed)

        # Initialize datasets
        self._setup_datasets()

        # Initialize models and optimizers
        self._setup_models()
        self._setup_optimizers()
        self._setup_proposal_functions()

        # Initialize metrics
        self._setup_metrics()

        # Training state
        self.step = 0
        self.start_time = time.time()

    def _setup_datasets(self):
        """Initialize training and test datasets."""
        SubjectLoader = self.config.get_dataset_class()

        self.train_dataset = SubjectLoader(
            subject_id=self.config.scene,
            root_fp=self.config.data_root,
            split=self.config.train_split,
            num_rays=self.config.training.init_batch_size,
            device=self.device,
            **self.config.scene_config.train_dataset_kwargs,
        )

        self.test_dataset = SubjectLoader(
            subject_id=self.config.scene,
            root_fp=self.config.data_root,
            split="test",
            num_rays=None,
            device=self.device,
            **self.config.scene_config.test_dataset_kwargs,
        )

    def _setup_models(self):
        """Initialize radiance field and proposal networks."""
        aabb = self.config.to_torch_aabb()

        # Create proposal networks
        self.proposal_networks = []
        for prop_config in self.config.model.proposal_networks_config:
            prop_net = NGPDensityField(
                aabb=aabb, unbounded=self.config.scene_config.unbounded, **prop_config
            ).to(self.device)
            self.proposal_networks.append(prop_net)

        # Create main radiance field
        self.radiance_field = NGPRadianceField(
            aabb=aabb, unbounded=self.config.scene_config.unbounded
        ).to(self.device)

    def _setup_optimizers(self):
        """Initialize optimizers and schedulers."""
        # Proposal network optimizer
        # self.prop_optimizer = torch.optim.Adam(
        #     itertools.chain(*[p.parameters() for p in self.proposal_networks]),
        #     lr=self.config.training.learning_rate,
        #     eps=self.config.training.eps,
        #     weight_decay=self.config.training.weight_decay,
        # )

        # Radiance field optimizer
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

        # self.prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        #     [
        #         torch.optim.lr_scheduler.LinearLR(
        #             self.prop_optimizer, start_factor=0.01, total_iters=100
        #         ),
        #         torch.optim.lr_scheduler.MultiStepLR(
        #             self.prop_optimizer, milestones=milestones, gamma=0.33
        #         ),
        #     ]
        # )

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

        # Estimator
        self.estimator = PropNetEstimator(self.prop_optimizer, self.prop_scheduler).to(
            self.device
        )

        # Gradient scaler
        self.grad_scaler = torch.cuda.amp.GradScaler(
            self.config.training.grad_scaler_init
        )

    def _setup_proposal_functions(self):
        """Setup proposal network utility functions."""
        self.proposal_requires_grad_fn = get_proposal_requires_grad_fn()

    def _setup_metrics(self):
        """Initialize metric computation tools."""
        self.lpips_net = LPIPS(net="vgg").to(self.device)
        self.lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
        self.lpips_fn = lambda x, y: self.lpips_net(
            self.lpips_norm_fn(x), self.lpips_norm_fn(y)
        ).mean()

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.

        Returns:
            Dictionary containing loss and metrics for this step
        """
        # Set models to training mode
        self.radiance_field.train()
        for p in self.proposal_networks:
            p.train()
        self.estimator.train()

        # Sample training data
        i = torch.randint(0, len(self.train_dataset), (1,)).item()
        data = self.train_dataset[i]

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
        rays = generate_camera_rays(x, y, c2w, self.train_dataset)

        # Determine if proposal networks need gradients
        proposal_requires_grad = self.proposal_requires_grad_fn(self.step)

        # Render
        rgb, acc, depth, extras = render_image_with_propnet(
            self.radiance_field,
            self.proposal_networks,
            self.estimator,
            rays,
            # rendering options
            num_samples=self.config.model.num_samples,
            num_samples_per_prop=self.config.model.num_samples_per_prop,
            near_plane=self.config.scene_config.near_plane,
            far_plane=self.config.scene_config.far_plane,
            sampling_type=self.config.model.sampling_type,
            opaque_bkgd=self.config.model.opaque_bkgd,
            render_bkgd=render_bkgd,
            # train options
            proposal_requires_grad=proposal_requires_grad,
        )

        # Update estimator
        self.estimator.update_every_n_steps(
            extras["trans"], proposal_requires_grad, loss_scaler=1024
        )

        # Compute loss
        loss = F.smooth_l1_loss(rgb, pixels)

        # Backward pass
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.optimizer.step()
        self.scheduler.step()

        # Compute metrics
        mse_loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(mse_loss) / torch.log(torch.tensor(10.0))

        self.step += 1

        return {
            "loss": loss.item(),
            "mse_loss": mse_loss.item(),
            "psnr": psnr.item(),
            "num_rays": len(pixels),
            "max_depth": depth.max().item(),
        }

    def should_print(self) -> bool:
        """Check if we should print training stats."""
        return self.step == 1 or self.step % self.config.training.print_every == 0

    def should_evaluate(self) -> bool:
        """Check if we should run evaluation."""
        return self.step > 0 and self.step % self.config.training.eval_every == 0

    def print_training_stats(self, metrics: Dict[str, float]):
        """Print training statistics."""
        elapsed_time = time.time() - self.start_time
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={self.step} | "
            f"loss={metrics['loss']:.5f} | psnr={metrics['psnr']:.2f} | "
            f"num_rays={metrics['num_rays']:d} | "
            f"max_depth={metrics['max_depth']:.3f}"
        )
