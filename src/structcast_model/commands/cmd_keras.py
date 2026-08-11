"""Keras related commands for the StructCast Model CLI application."""

from time import time
from typing import TYPE_CHECKING, Any

from structcast.utils.base import configure_security
from typer import Argument, Option, Typer

from structcast_model.commands.utils import (
    bool_or_path_or_dict_parser,
    dict_parser,
    instantiate_object,
    path_or_any_parser,
    reduce_dict,
    tensor_shape_parser,
)

if TYPE_CHECKING:
    import jax
    from structcast.core import instantiator

    import keras
    from structcast_model.builders import keras_builder
    from structcast_model.keras import trainer as keras_trainer
    import torch
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    jax = LazyModuleImporter("jax")
    instantiator = LazyModuleImporter("structcast.core.instantiator")
    keras = LazyModuleImporter("keras")
    keras_builder = LazyModuleImporter("structcast_model.builders.keras_builder")
    keras_trainer = LazyModuleImporter("structcast_model.keras.trainer")
    torch = LazyModuleImporter("torch")


app = Typer(no_args_is_help=True)
creator = Typer(no_args_is_help=True)
app.add_typer(creator, name="create", help="Commands for creating Keras layers.")

template_param = Option(
    None,
    "--parameter",
    "-p",
    parser=dict_parser,
    help="Parameters to format the template configuration file with. "
    'Each parameter should be in the format of "key: {...}", where `key` is the name of the parameter group, '
    "and the value is a dictionary of keyword arguments for formatting the template. "
    'For example: --parameter "model: {input_size: 128, output_size: 10}"',
)
output_script_path = Option(None, "--output", "-o", help="Output script path (Python).")
model_pattern = Argument(
    parser=path_or_any_parser,
    help="The object pattern used to instantiate models. "
    "For example, if the model is defined as `my_package.MyModel(...)`, "
    'then the pattern should be "[_obj_, {_addr_: my_package.MyModel, _file_: my_package.py}, {_call_: {...}}]" or '
    '"[_obj_, [_addr_, my_package.MyModel, my_package.py], {_call_: {...}}]".',
)
shapes = Option(
    None,
    "--shape",
    "-s",
    parser=tensor_shape_parser,
    help="Input tensor shapes as a dictionary, e.g., 'image: [224, 224, 3]'.",
)
device = Option(
    None,
    "--device",
    "-d",
    help="The device to run the inference time measurement on. "
    "If not specified, the first available device will be used.",
)
compile_pattern: dict[str, Any] | None = Option(
    None,
    "--compile",
    "-c",
    parser=bool_or_path_or_dict_parser,
    help='Whether to compile the model using "keras.Model.compile". '
    'Can be set to true/false, a path to a YAML file, or a dictionary of keyword arguments for "keras.Model.compile".',
)


@creator.command(name="model")
def create_model(
    cfg_path: str = Argument(..., help="Path to the model configuration file."),
    output: str | None = output_script_path,
    parameters: list[dict] | None = template_param,
    classname: str = Option("Model", "--classname", "-c", help="Name the model class."),
    structured_output: bool = Option(True, help="Enable structured output for the model."),
    sublayer: str | None = Option(
        None, "--sublayer", "-s", help="The reference to a sublayer in the template to build instead of the root layer."
    ),
) -> None:
    """Create a Keras layer from the given configuration file and parameters."""
    keras_builder.KerasBuilder.from_path(cfg_path)(
        parameters=reduce_dict(parameters),
        classname=classname,
        forced_structured_output=structured_output,
        user_defined_layer=sublayer,
    )(output)


def _get_sync_fn(device: str) -> Any:
    """Return a synchronization function appropriate for the current Keras backend."""
    backend = keras.backend.backend()
    if backend == "jax":
        return lambda xs: jax.tree_util.tree_map(lambda x: x.block_until_ready(), xs)
    if backend == "torch" and "gpu" in device:
        return lambda _: torch.cuda.synchronize()
    return lambda _: None


@app.command(name="time")
def measure_inference_time(
    model_pattern: Any = model_pattern,
    shapes: dict | None = shapes,
    device: str | None = device,
    compile_pattern: dict[str, Any] | None = compile_pattern,
    training_mode: bool = Option(
        False,
        help="Whether to set the model to training mode during inference time measurement. "
        "This can affect the inference time due to differences in behavior (e.g., dropout, batch norm).",
    ),
    warmup_runs: int = Option(2, "--warmup-runs", "-w", help="Number of warmup runs before measuring inference time."),
    times: int = Option(10, "--times", "-t", help="Number of iterations to measure the inference time."),
    batch_size: int = Option(
        1, "--batch-size", "-b", help="Batch size for the input tensors during inference time measurement."
    ),
) -> None:
    """Measure the average inference time of a Keras model."""
    configure_security()
    device = keras_trainer.get_keras_device(device)
    print("Initializing the model...")
    model = instantiate_object(model_pattern)
    shapes = keras_trainer.resolve_input_shapes(model, shapes)
    model = keras_trainer.initial_model(model, shapes)
    if compile_pattern is None:
        print("Skipping compilation...")
    else:
        print("Compiling the model...")
        model.compile(optimizer=None, **instantiator.instantiate(compile_pattern))
    sync = _get_sync_fn(device)

    def _measure_single_run() -> float:
        inputs = keras_trainer.create_numpy_inputs(shapes, batch_size=batch_size)
        start_time = time()
        sync(model(inputs, training=training_mode))
        return time() - start_time

    print(f"Running {warmup_runs} warmup runs...")
    for _ in range(warmup_runs):
        _measure_single_run()
    elapsed_time = 0.0
    for ind in range(times):
        print(f"Running inference iteration {ind + 1}/{times}...")
        elapsed_time += _measure_single_run()
    mode_str = "training" if training_mode else "evaluation"
    print(f'Average inference time over {times} runs ("{mode_str}" mode): {elapsed_time / times:.6f} seconds.')


__all__ = ["app"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
