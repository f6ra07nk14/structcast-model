"""Types for torch module."""

from typing import TypeAlias

from torch import Tensor as _Tensor, device as _device, dtype as _dtype

DeviceLike: TypeAlias = _device | str
"""Device like type."""

DType: TypeAlias = _dtype
"""Data type."""

Tensor: TypeAlias = _Tensor
"""Tensor type."""
