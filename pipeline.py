"""
NeRF training pipeline that orchestrates the entire training and evaluation process.
"""

from typing import Dict, Any

import torch

from config import NeRFConfig

# from trainers.trainer import NeRFTrainer
# from trainers.MLEtrainer import MLETrainer
from trainers.BAtrainer import BATrainer
from trainers.BAevaluator import BAEvaluator


class NeRFPipeline:
    """
    Complete NeRF training and evaluation pipeline.
    """

    def __init__(self, config: NeRFConfig):
        """
        Initialize the NeRF pipeline.

        Args:
            config: NeRFConfig containing all training parameters
        """
        self.config = config

        # Initialize trainer and evaluator
        self.trainer = BATrainer(config)
        # self.trainer = MLETrainer(config)
        self.evaluator = BAEvaluator(self.trainer, save_images=True)

        # Training history
        self.training_history = []
        self.evaluation_history = []

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the complete training loop.

        Args:
            verbose: Whether to print training progress

        Returns:
            Dictionary containing training and evaluation results
        """
        print(f"Starting training for {self.config.training.max_steps} steps...")
        print(f"Scene: {self.config.scene}")
        print(f"Device: {self.config.device}")

        while self.trainer.step <= self.config.training.max_steps:
            # Training step
            metrics = self.trainer.train_step()

            # Log training metrics
            with torch.no_grad():
                if self.trainer.should_print() and verbose:
                    self.trainer.print_training_stats(metrics)

                # Store metrics
                metrics["step"] = self.trainer.step
                self.training_history.append(metrics)

                # Evaluation
                if self.trainer.should_evaluate():
                    print("Running evaluation...")
                    eval_results = self.evaluator.evaluate(verbose=verbose)
                    eval_results["step"] = self.trainer.step
                    self.evaluation_history.append(eval_results)

                    if verbose:
                        self.evaluator.print_evaluation_results(eval_results)

        # Final evaluation
        print("Training completed. Running final evaluation...")
        final_eval = self.evaluator.evaluate(verbose=verbose)
        final_eval["step"] = self.trainer.step
        self.evaluation_history.append(final_eval)

        if verbose:
            print("\nFinal Results:")
            self.evaluator.print_evaluation_results(final_eval)

        return {
            "final_evaluation": final_eval,
            "training_history": self.training_history,
            "evaluation_history": self.evaluation_history,
        }
