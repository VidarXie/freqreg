"""
Configuration classes for NeRF training pipeline.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch
from utils.render_utils import MIPNERF360_UNBOUNDED_SCENES, NERF_SYNTHETIC_SCENES


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""

    max_steps: int
    init_batch_size: int
    weight_decay: float
    learning_rate: float = 1e-2
    eps: float = 1e-15
    grad_scaler_init: int = 2**10
    print_every: int = None
    eval_every: int = None  # If None, evaluate only at the end

    def __post_init__(self):
        if self.print_every is None:
            self.print_every = self.max_steps // 100
        if self.eval_every is None:
            self.eval_every = self.max_steps // 5


@dataclass
class SceneConfig:
    """Scene-specific parameters configuration."""

    unbounded: bool
    aabb: List[float]
    near_plane: float
    far_plane: float
    train_dataset_kwargs: Dict[str, Any]
    test_dataset_kwargs: Dict[str, Any]


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    proposal_networks_config: List[Dict[str, Any]]
    num_samples: int
    num_samples_per_prop: List[int]
    sampling_type: str
    opaque_bkgd: bool


@dataclass
class NeRFConfig:
    """Complete NeRF training configuration."""

    # Basic settings
    scene: str
    data_root: str
    train_split: str = "train"
    test_chunk_size: int = 8192
    device: str = "cuda:0"
    seed: int = 42

    # Configurations
    training: Optional[TrainingConfig] = None
    scene_config: Optional[SceneConfig] = None
    model: Optional[ModelConfig] = None

    def __post_init__(self):
        """Auto-configure based on scene type if not provided."""
        if self.training is None or self.scene_config is None or self.model is None:
            self._auto_configure()

    def _auto_configure(self):
        """Automatically configure based on scene type."""
        if self.scene in MIPNERF360_UNBOUNDED_SCENES:
            self._configure_mipnerf360()
        elif self.scene in NERF_SYNTHETIC_SCENES:
            self._configure_nerf_synthetic()
        else:
            raise ValueError(f"Unknown scene: {self.scene}")

    def _configure_mipnerf360(self):
        """Configure for MipNeRF360 scenes."""
        self.training = TrainingConfig(
            max_steps=100000, init_batch_size=8192, weight_decay=0.0
        )

        self.scene_config = SceneConfig(
            unbounded=True,
            aabb=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            near_plane=0.2,
            far_plane=1e3,
            train_dataset_kwargs={"color_bkgd_aug": "random", "factor": 4},
            test_dataset_kwargs={"factor": 4},
        )

        self.model = ModelConfig(
            proposal_networks_config=[
                {"n_levels": 5, "max_resolution": 128},
                {"n_levels": 5, "max_resolution": 256},
            ],
            num_samples=48,
            num_samples_per_prop=[256, 96],
            sampling_type="lindisp",
            opaque_bkgd=True,
        )

        self.output_dir = f"./output/{self.scene}_{self.seed}"

    def _configure_nerf_synthetic(self):
        """Configure for NeRF synthetic scenes."""
        weight_decay = 1e-5 if self.scene in ["materials", "ficus", "drums"] else 1e-6

        self.training = TrainingConfig(
            max_steps=20000, init_batch_size=8192, weight_decay=weight_decay
        )

        self.scene_config = SceneConfig(
            unbounded=False,
            aabb=[-1.5, -1.5, -1.5, 1.5, 1.5, 1.5],
            near_plane=2.0,
            far_plane=6.0,
            train_dataset_kwargs={},
            test_dataset_kwargs={},
        )

        self.model = ModelConfig(
            proposal_networks_config=[{"n_levels": 5, "max_resolution": 128}],
            num_samples=64,
            num_samples_per_prop=[128],
            sampling_type="uniform",
            opaque_bkgd=False,
        )

        self.output_dir = f"./output/{self.scene}_{self.seed}"

    def get_dataset_class(self):
        """Get appropriate dataset class based on scene type."""
        if self.scene in MIPNERF360_UNBOUNDED_SCENES:
            from datasets.nerf_360_v2 import SubjectLoader
        else:
            from datasets.nerf_synthetic import SubjectLoader
        return SubjectLoader

    def to_torch_aabb(self) -> torch.Tensor:
        """Convert AABB to torch tensor on device."""
        return torch.tensor(self.scene_config.aabb, device=self.device)
