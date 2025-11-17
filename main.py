#!/usr/bin/env python3
"""
Clean and modular NeRF training script.
This script provides a simple interface to train NeRF models with proposal networks.

Usage:
    python train_nerf_modular.py --scene lego --data_root data/nerf_synthetic
    python train_nerf_modular.py --scene flowers --data_root data/360_v2
"""

import argparse

from config import NeRFConfig
from pipeline import NeRFPipeline
from utils.render_utils import (
    NERF_SYNTHETIC_SCENES,
    MIPNERF360_UNBOUNDED_SCENES,
)


def create_config_from_args(args) -> NeRFConfig:
    """Create NeRFConfig from command line arguments."""
    config = NeRFConfig(
        scene=args.scene,
        data_root=args.data_root,
        exp_name=args.exp_name,
        task=args.task,
        train_split=args.train_split,
        test_chunk_size=args.test_chunk_size,
        device=args.device,
        seed=args.seed,
        noise_std=args.noise_std,
    )

    # Override training parameters if specified
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.batch_size is not None:
        config.training.init_batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Modular NeRF Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--scene",
        type=str,
        help="Scene to train on",
        choices=NERF_SYNTHETIC_SCENES + MIPNERF360_UNBOUNDED_SCENES,
    )
    parser.add_argument("--data_root", type=str, help="Root directory of the dataset")
    parser.add_argument("--exp_name", type=str, help="Experiment name for logging")
    parser.add_argument(
        "--task", type=str, default="ba", choices=["ba", "fs", "nerf"], help="Task type"
    )
    parser.add_argument("--noise_std", type=float, help="Noise standard deviation")

    # Optional training arguments
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        choices=["train", "trainval"],
        help="Training split to use",
    )
    parser.add_argument(
        "--test_chunk_size",
        type=int,
        default=8192,
        help="Chunk size for test rendering",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Training hyperparameters
    parser.add_argument(
        "--max_steps", type=int, help="Maximum training steps (overrides default)"
    )
    parser.add_argument("--batch_size", type=int, help="Batch size (overrides default)")
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate (overrides default)"
    )

    # Logging
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Verbose output"
    )
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Handle verbosity
    verbose = args.verbose and not args.quiet

    # Create configuration
    config = create_config_from_args(args)

    # Print configuration
    if verbose:
        print("=" * 50)
        print("NeRF Training Configuration")
        print("=" * 50)
        print(f"Scene: {config.scene}")
        print(f"Data root: {config.data_root}")
        print(f"Train split: {config.train_split}")
        print(f"Device: {config.device}")
        print(f"Max steps: {config.training.max_steps}")
        print(f"Batch size: {config.training.init_batch_size}")
        print(f"Learning rate: {config.training.learning_rate}")
        print("=" * 50)

    # Initialize pipeline
    pipeline = NeRFPipeline(config=config)

    _ = pipeline.train(verbose=verbose)

    if verbose:
        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print("=" * 50)


if __name__ == "__main__":
    main()
