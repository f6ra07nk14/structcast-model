from dataclasses import dataclass

import torch.nn

from structcast_model.base_trainer import BaseTrainer


@dataclass(kw_only=True)
class TorchTrainer(BaseTrainer[torch.nn.Module]):
    """Trainer for PyTorch models."""
