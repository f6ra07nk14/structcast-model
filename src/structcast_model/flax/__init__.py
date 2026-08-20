"""Flax module for StructCast-Model."""

from typing import TYPE_CHECKING

__all__ = [
    "AXIS",
    "PRESET_RULES",
    "TACTICS",
    "FlaxBestCriterion",
    "FlaxDistributedStrategy",
    "FlaxTracker",
    "FlaxTrainer",
    "FlaxTrainingStateSaver",
    "TensorInitializer",
    "create_jax_inputs",
    "distributed",
    "get_jax_device",
    "get_jax_devices",
    "get_learning_rate",
    "layers",
    "no_weight_decay_mask",
    "optimizers",
    "resolve_input_shapes",
    "restore_training_state",
    "trainer",
    "unwrap_variables",
]

if TYPE_CHECKING:
    from structcast_model.flax import layers
    from structcast_model.flax.distributed import AXIS, PRESET_RULES, TACTICS, FlaxDistributedStrategy
    from structcast_model.flax.optimizers import get_learning_rate, no_weight_decay_mask, unwrap_variables
    from structcast_model.flax.trainer import (
        FlaxBestCriterion,
        FlaxTracker,
        FlaxTrainer,
        FlaxTrainingStateSaver,
        TensorInitializer,
        create_jax_inputs,
        get_jax_device,
        get_jax_devices,
        resolve_input_shapes,
        restore_training_state,
    )
else:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    # Each symbol is listed exactly once: _class_to_module is a dict comprehension, so a name listed
    # twice silently keeps the last writer. A re-exported name goes under its defining module when
    # that module has an entry of its own -- distributed re-exports get_jax_device, routed to trainer
    # instead. The layers subpackage stays submodule-only, as its torch twin does.
    import_structure = {
        "distributed": ["AXIS", "PRESET_RULES", "TACTICS", "FlaxDistributedStrategy"],
        "layers": [],
        "optimizers": ["get_learning_rate", "no_weight_decay_mask", "unwrap_variables"],
        "trainer": [
            "FlaxBestCriterion",
            "FlaxTracker",
            "FlaxTrainer",
            "FlaxTrainingStateSaver",
            "TensorInitializer",
            "create_jax_inputs",
            "get_jax_device",
            "get_jax_devices",
            "resolve_input_shapes",
            "restore_training_state",
        ],
    }
    sys.modules[__name__] = LazySelectedImporter(__name__, globals(), import_structure)
