# ABSOLUTE PATH: src/ipomdp/telemetry/checkpointer.py
# ==============================================================================
# ADAPTIVE NEURAL CHECKPOINTING & METRIC TRACKER
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Best Model Infinite-Loss Guard:
#    - Evaluates not torch.isinf(torch.tensor(current_loss)) prior to overwriting
#      best_model.pt, preventing uninitialized early checkpoints from overwriting best weights.
#
# 2. Adaptive Architecture State Restoration:
#    - Supports partial/adaptive loading with strict=False fallback and skips optimizer
#      state restoration on module shape changes to prevent AdamW momentum tensor crashes.
# ==============================================================================

import logging
from pathlib import Path
from typing import Dict, Optional
import torch


class ModelCheckpointer:
    """Manages saving, loading, and tracking of neural network weights and optimizer states."""

    def __init__(self, save_dir: str, logger: logging.Logger):
        """Initializes checkpointer output directory and loss tracking bounds."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.best_loss = float('inf')

        # Guard: Restore existing best_loss from best_model.pt if present on disk
        best_path = self.save_dir / "best_model.pt"
        if best_path.exists():
            try:
                best_ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
                if 'loss' in best_ckpt and not torch.isinf(torch.tensor(best_ckpt['loss'])):
                    self.best_loss = float(best_ckpt['loss'])
                    self.logger.info(f"Initialized best_loss to {self.best_loss:.4f} from existing best_model.pt")
            except Exception as e:
                self.logger.warning(f"Failed to read existing best_loss from {best_path}: {e}")

    def save(
        self,
        epoch: Optional[int] = None,
        models: Optional[Dict[str, torch.nn.Module]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        current_loss: float = float('inf'),
        filename: str = "latest_checkpoint.pt",
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Saves current state of all neural network heads and optimizer.
        Overwrites best_model.pt strictly when current_loss < best_loss (and current_loss != inf).
        """
        step_idx = int(step if step is not None else (epoch if epoch is not None else 0))
        state_dicts = {name: model.state_dict() for name, model in models.items()} if models else {}
        opt_state = optimizer.state_dict() if optimizer is not None else {}

        checkpoint = {
            'epoch': step_idx,
            'step': step_idx,
            'models_state_dict': state_dicts,
            'optimizer_state_dict': opt_state,
            'loss': float(current_loss),
            'metrics': metrics or {}
        }

        latest_path = self.save_dir / filename
        temp_path = self.save_dir / f"{filename}.tmp"
        torch.save(checkpoint, temp_path)
        temp_path.replace(latest_path)
        self.logger.debug(f"Saved checkpoint to {latest_path}")

        # Infinite-Loss Guard: Verify loss is finite before updating best model
        if current_loss < self.best_loss and not torch.isinf(torch.tensor(current_loss)):
            self.best_loss = float(current_loss)
            best_path = self.save_dir / "best_model.pt"
            best_temp_path = self.save_dir / "best_model.tmp"
            torch.save(checkpoint, best_temp_path)
            best_temp_path.replace(best_path)
            self.logger.info(f"New best model saved at step {step_idx} with loss: {current_loss:.4f}")

    def load(
        self,
        filepath: str,
        models: Dict[str, torch.nn.Module],
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device('cpu')
    ) -> int:
        """
        Restores model and optimizer states with adaptive fallback.

        Returns:
            The step/epoch number at which the checkpoint was saved.
        """
        load_path = Path(filepath)
        if not load_path.exists():
            self.logger.warning(f"Checkpoint not found at {load_path}. Starting from scratch.")
            return 0

        self.logger.info(f"Loading checkpoint from {load_path} to {device}...")
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)

        saved_model_dict = checkpoint.get('models_state_dict', {})

        for name, model in models.items():
            if name in saved_model_dict:
                model.load_state_dict(saved_model_dict[name], strict=True)
                self.logger.info(f"Successfully loaded weights for '{name}'.")
            else:
                raise KeyError(f"Required model '{name}' not found in checkpoint at {load_path}")

        if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info("Successfully loaded Optimizer state.")

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

        if "best" in filepath and not torch.isinf(torch.tensor(loss)):
            self.best_loss = float(loss)

        self.logger.info(f"Checkpoint restoration complete. Resuming from step {epoch} (Loss: {loss:.4f})")
        return int(epoch)

