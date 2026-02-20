"""Types for torch module."""

from typing import TypeAlias

from torch import Tensor as _Tensor, device as _device, dtype as _dtype

DeviceLike: TypeAlias = _device | str
"""Device like type."""

DType = _dtype
"""Data type."""

Tensor = _Tensor
"""Tensor type."""
