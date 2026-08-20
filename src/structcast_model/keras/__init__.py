"""Keras module for StructCast-Model."""

from typing import TYPE_CHECKING

__all__ = [
    "AdapterSegment",
    "BackendAdapter",
    "Flow",
    "InferenceFlow",
    "JaxAdapter",
    "KerasBestCriterion",
    "KerasTracker",
    "KerasTrainer",
    "KerasTrainingStateSaver",
    "TensorFlowAdapter",
    "TensorInitializer",
    "TorchAdapter",
    "adapters",
    "collect_state_dict",
    "create_keras_inputs",
    "create_numpy_inputs",
    "get_keras_device",
    "initial_model",
    "layers",
    "resolve_input_shapes",
    "select_backend_adapter",
    "trainer",
]

if TYPE_CHECKING:
    from structcast_model.keras import adapters, layers, trainer
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
    from structcast_model.keras.trainer import (
        KerasBestCriterion,
        KerasTracker,
        KerasTrainer,
        KerasTrainingStateSaver,
        TensorInitializer,
        collect_state_dict,
        create_keras_inputs,
        create_numpy_inputs,
        get_keras_device,
        initial_model,
        resolve_input_shapes,
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
        "layers": [],
        "trainer": [
            "KerasBestCriterion",
            "KerasTracker",
            "KerasTrainer",
            "KerasTrainingStateSaver",
            "TensorInitializer",
            "collect_state_dict",
            "create_keras_inputs",
            "create_numpy_inputs",
            "get_keras_device",
            "initial_model",
            "resolve_input_shapes",
        ],
    }
    sys.modules[__name__] = LazySelectedImporter(__name__, globals(), import_structure)
