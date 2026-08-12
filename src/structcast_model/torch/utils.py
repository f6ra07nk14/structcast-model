"""Shared PyTorch device helpers."""

from logging import getLogger
from typing import TYPE_CHECKING

import torch

logger = getLogger(__name__)


def get_torch_device(device: str | None = None) -> str:
    """Get the device to run the model on."""
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if "cpu" in device:
        return device
    if "cuda" in device:
        if torch.cuda.is_available():
            return device
        logger.warning("CUDA is not available. Using CPU instead.")
        return "cpu"
    raise ValueError(f'Only "cpu" and "cuda" (with optional rank suffix) are supported. Got invalid device: {device}')


def get_torch_device_type(device: str | None = None) -> str:
    """Get the device type (cpu or cuda) from the device string."""
    return get_torch_device(device).split(":")[0]


__all__ = ["get_torch_device", "get_torch_device_type"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
