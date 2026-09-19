"""Keras module for StructCast-Model."""

from typing import TYPE_CHECKING

__all__ = [
    "AXIS",
    "MODEL_AXIS",
    "PRESET_RULES",
    "REJECTED",
    "TACTICS",
    "AdapterSegment",
    "BackendAdapter",
    "Flow",
    "InferenceFlow",
    "JaxAdapter",
    "KerasBestCriterion",
    "KerasDistributedStrategy",
    "KerasTracker",
    "KerasTrainer",
    "KerasTrainingStateSaver",
    "RuleModelParallel",
    "TensorFlowAdapter",
    "TensorInitializer",
    "TorchAdapter",
    "apply_state_dict",
    "collect_state_dict",
    "create_keras_inputs",
    "create_numpy_inputs",
    "get_keras_device",
    "initial_model",
    "layers",
    "resolve_input_shapes",
    "restore_training_state",
    "select_backend_adapter",
    "swap_ema_weights",
]

if TYPE_CHECKING:
    from structcast_model.keras import layers
    from structcast_model.keras.adapters import (
        AdapterSegment,
        BackendAdapter,
        Flow,
        InferenceFlow,
        JaxAdapter,
        TensorFlowAdapter,
        TorchAdapter,
        select_backend_adapter,
        swap_ema_weights,
    )
    from structcast_model.keras.distributed import (
        AXIS,
        MODEL_AXIS,
        PRESET_RULES,
        REJECTED,
        TACTICS,
        KerasDistributedStrategy,
        RuleModelParallel,
    )
    from structcast_model.keras.trainer import (
        KerasBestCriterion,
        KerasTracker,
        KerasTrainer,
        KerasTrainingStateSaver,
        TensorInitializer,
        create_keras_inputs,
        create_numpy_inputs,
        initial_model,
        resolve_input_shapes,
        restore_training_state,
    )
    from structcast_model.keras.utils import apply_state_dict, collect_state_dict, get_keras_device
else:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    # Each symbol is listed exactly once: _class_to_module is a dict comprehension, so a name listed
    # twice silently keeps the last writer. A re-exported name goes under its defining module when
    # that module has an entry of its own -- distributed re-exports the utils helpers, routed to
    # utils instead. The layers subpackage stays submodule-only, as its flax and torch twins do.
    import_structure = {
        "adapters": [
            "AdapterSegment",
            "BackendAdapter",
            "Flow",
            "InferenceFlow",
            "JaxAdapter",
            "TensorFlowAdapter",
            "TorchAdapter",
            "select_backend_adapter",
            "swap_ema_weights",
        ],
        "distributed": [
            "AXIS",
            "MODEL_AXIS",
            "PRESET_RULES",
            "REJECTED",
            "TACTICS",
            "KerasDistributedStrategy",
            "RuleModelParallel",
        ],
        "layers": [],
        "trainer": [
            "KerasBestCriterion",
            "KerasTracker",
            "KerasTrainer",
            "KerasTrainingStateSaver",
            "TensorInitializer",
            "create_keras_inputs",
            "create_numpy_inputs",
            "initial_model",
            "resolve_input_shapes",
            "restore_training_state",
        ],
        "utils": ["apply_state_dict", "collect_state_dict", "get_keras_device"],
    }
    sys.modules[__name__] = LazySelectedImporter(__name__, globals(), import_structure)
