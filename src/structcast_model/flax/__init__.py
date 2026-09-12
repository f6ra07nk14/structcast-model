"""Flax module for StructCast-Model."""

from typing import TYPE_CHECKING

__all__ = [
    "AXIS",
    "MODEL_AXIS",
    "PRESET_RULES",
    "TACTICS",
    "TP_PRESETS",
    "FlaxBestCriterion",
    "FlaxDistributedStrategy",
    "FlaxTracker",
    "FlaxTrainer",
    "FlaxTrainingStateSaver",
    "ShardedDataset",
    "TensorInitializer",
    "create_jax_inputs",
    "donate_argnames",
    "dot_general_out",
    "get_jax_device",
    "get_jax_devices",
    "get_learning_rate",
    "layers",
    "no_weight_decay_mask",
    "resolve_input_shapes",
    "restore_training_state",
    "unwrap_variables",
]

if TYPE_CHECKING:
    from structcast_model.flax import layers
    from structcast_model.flax.distributed import (
        AXIS,
        MODEL_AXIS,
        PRESET_RULES,
        TACTICS,
        TP_PRESETS,
        FlaxDistributedStrategy,
    )
    from structcast_model.flax.optimizers import get_learning_rate, no_weight_decay_mask, unwrap_variables
    from structcast_model.flax.trainer import (
        FlaxBestCriterion,
        FlaxTracker,
        FlaxTrainer,
        FlaxTrainingStateSaver,
        ShardedDataset,
        TensorInitializer,
        create_jax_inputs,
        resolve_input_shapes,
        restore_training_state,
    )
    from structcast_model.flax.utils import donate_argnames, dot_general_out, get_jax_device, get_jax_devices
else:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    # Each symbol is listed exactly once: _class_to_module is a dict comprehension, so a name listed
    # twice silently keeps the last writer. A re-exported name goes under its defining module when
    # that module has an entry of its own -- distributed re-exports get_jax_device, routed to utils
    # instead. The layers subpackage stays submodule-only, as its torch twin does.
    import_structure = {
        "distributed": ["AXIS", "MODEL_AXIS", "PRESET_RULES", "TACTICS", "TP_PRESETS", "FlaxDistributedStrategy"],
        "layers": [],
        "optimizers": ["get_learning_rate", "no_weight_decay_mask", "unwrap_variables"],
        "trainer": [
            "FlaxBestCriterion",
            "FlaxTracker",
            "FlaxTrainer",
            "FlaxTrainingStateSaver",
            "ShardedDataset",
            "TensorInitializer",
            "create_jax_inputs",
            "resolve_input_shapes",
            "restore_training_state",
        ],
        "utils": ["donate_argnames", "dot_general_out", "get_jax_device", "get_jax_devices"],
    }
    sys.modules[__name__] = LazySelectedImporter(__name__, globals(), import_structure)
