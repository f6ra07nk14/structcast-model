"""Torch Extensions for StructCast-Model."""

from typing import TYPE_CHECKING

__all__ = [
    "CriteriaTracker",
    "DType",
    "DeviceLike",
    "DistributedDataParallelStrategy",
    "DistributedStrategy",
    "FullyShardedDataParallelStrategy",
    "SingleDeviceStrategy",
    "Tensor",
    "TensorInitializer",
    "TorchBestCriterion",
    "TorchTracker",
    "TorchTrainer",
    "TrainingStateSaver",
    "autocast_inputs",
    "create_opt",
    "create_torch_inputs",
    "distributed",
    "get_decays",
    "get_learning_rate",
    "get_named_parameters",
    "get_param_groups",
    "get_torch_device",
    "get_torch_device_type",
    "initial_distributed_env",
    "initial_model",
    "layers",
    "matched_shard_modules",
    "optimizers",
    "resolve_input_shapes",
    "restore_requires_grad",
    "set_lr_scale",
    "sync_gate",
    "trainer",
    "types",
    "utils",
]

if TYPE_CHECKING:
    from structcast_model.torch import layers
    from structcast_model.torch.distributed import (
        DistributedDataParallelStrategy,
        DistributedStrategy,
        FullyShardedDataParallelStrategy,
        SingleDeviceStrategy,
        initial_distributed_env,
        matched_shard_modules,
        sync_gate,
    )
    from structcast_model.torch.optimizers import (
        create_opt,
        get_decays,
        get_learning_rate,
        get_named_parameters,
        get_param_groups,
        restore_requires_grad,
        set_lr_scale,
    )
    from structcast_model.torch.trainer import (
        CriteriaTracker,
        TorchBestCriterion,
        TorchTracker,
        TorchTrainer,
        TrainingStateSaver,
        autocast_inputs,
        create_torch_inputs,
        initial_model,
        resolve_input_shapes,
    )
    from structcast_model.torch.types import DeviceLike, DType, Tensor, TensorInitializer
    from structcast_model.torch.utils import get_torch_device, get_torch_device_type
else:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    # Each symbol is listed exactly once: _class_to_module is a dict comprehension, so a name listed
    # twice silently keeps the last writer. A re-exported name goes under its defining module when
    # that module has an entry of its own -- trainer re-exports initial_distributed_env,
    # get_torch_device and get_torch_device_type, routed to distributed/utils instead. CriteriaTracker
    # is defined in the layers subpackage, whose entry stays submodule-only, so trainer routes it.
    import_structure = {
        "distributed": [
            "DistributedDataParallelStrategy",
            "DistributedStrategy",
            "FullyShardedDataParallelStrategy",
            "SingleDeviceStrategy",
            "initial_distributed_env",
            "matched_shard_modules",
            "sync_gate",
        ],
        "layers": [],
        "optimizers": [
            "create_opt",
            "get_decays",
            "get_learning_rate",
            "get_named_parameters",
            "get_param_groups",
            "restore_requires_grad",
            "set_lr_scale",
        ],
        "trainer": [
            "CriteriaTracker",
            "TorchBestCriterion",
            "TorchTracker",
            "TorchTrainer",
            "TrainingStateSaver",
            "autocast_inputs",
            "create_torch_inputs",
            "initial_model",
            "resolve_input_shapes",
        ],
        "types": ["DType", "DeviceLike", "Tensor", "TensorInitializer"],
        "utils": ["get_torch_device", "get_torch_device_type"],
    }
    sys.modules[__name__] = LazySelectedImporter(__name__, globals(), import_structure)
