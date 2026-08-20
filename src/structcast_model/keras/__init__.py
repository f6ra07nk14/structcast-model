"""Keras module for StructCast-Model."""

from typing import TYPE_CHECKING

__all__ = [
    "AXIS",
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
    "adapters",
    "apply_state_dict",
    "collect_state_dict",
    "create_keras_inputs",
    "create_numpy_inputs",
    "distributed",
    "get_keras_device",
    "initial_model",
    "layers",
    "resolve_input_shapes",
    "restore_training_state",
    "select_backend_adapter",
    "trainer",
]

if TYPE_CHECKING:
    from structcast_model.keras import adapters, distributed, layers, trainer
    from structcast_model.keras.adapters import (
        AdapterSegment,
        BackendAdapter,
        Flow,
        InferenceFlow,
        JaxAdapter,
        TensorFlowAdapter,
        TorchAdapter,
        select_backend_adapter,
    )
    from structcast_model.keras.distributed import (
        AXIS,
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
        apply_state_dict,
        collect_state_dict,
        create_keras_inputs,
        create_numpy_inputs,
        get_keras_device,
        initial_model,
        resolve_input_shapes,
        restore_training_state,
    )
else:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    # Each symbol is listed exactly once: _class_to_module is a dict comprehension, so a name listed
    # twice silently keeps the last writer. The layers subpackage stays submodule-only, as its flax
    # and torch twins do.
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
        ],
        "distributed": ["AXIS", "PRESET_RULES", "REJECTED", "TACTICS", "KerasDistributedStrategy", "RuleModelParallel"],
        "layers": [],
        "trainer": [
            "KerasBestCriterion",
            "KerasTracker",
            "KerasTrainer",
            "KerasTrainingStateSaver",
            "TensorInitializer",
            "apply_state_dict",
            "collect_state_dict",
            "create_keras_inputs",
            "create_numpy_inputs",
            "get_keras_device",
            "initial_model",
            "resolve_input_shapes",
            "restore_training_state",
        ],
    }
    sys.modules[__name__] = LazySelectedImporter(__name__, globals(), import_structure)
