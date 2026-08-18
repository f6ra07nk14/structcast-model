"""Keras related commands for the StructCast Model CLI application."""

from time import time
from typing import TYPE_CHECKING, Any

from typer import Argument, Option, Typer

from structcast_model.commands.shared_args import (
    batch_size,
    model_pattern,
    output_script_path,
    shapes_help,
    template_param_option,
    times,
    training_mode,
    warmup_runs,
)
from structcast_model.commands.utils import (
    bool_or_path_or_dict_parser,
    instantiate_object,
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
app.add_typer(creator, name="create", help="Commands for creating Keras layer classes.")

template_param = template_param_option('For example: --parameter "model: {input_size: 128, output_size: 10}"')
shapes = Option(
    None,
    "--shape",
    "-s",
    parser=tensor_shape_parser,
    help=shapes_help('"image: [224, 224, 3]"', "numpy.zeros")
    + " Omit it only when the model declares INPUT_SHAPES itself.",
)
device = Option(
    None,
    "--device",
    "-d",
    help='Device to time on, named as returned by "keras.distribution.list_devices()", e.g. "cpu:0" or "gpu:0"; '
    "an unavailable name aborts. If not specified, the first listed device is used. It does not place the model or "
    "the inputs; the Keras backend decides where the computation runs. The name only changes how the timing loop "
    'synchronizes on the torch backend, where a name containing "gpu" selects a CUDA sync.',
)
compile_pattern: dict[str, Any] | None = Option(
    None,
    "--compile",
    "-c",
    parser=bool_or_path_or_dict_parser,
    help='Whether to configure the model with "keras.Model.compile" (Keras run configuration, e.g. jit_compile, not '
    "graph compilation). Omitted or false skips the call; true calls it with defaults. Can also be a path to an "
    'existing YAML/JSON file, or a dictionary of keyword arguments for "keras.Model.compile"; "optimizer" is always '
    "passed as None and cannot be set here.",
)


@creator.command(name="model")
def create_model(
    cfg_path: str = Argument(..., help="Path to the model configuration file."),
    output: str | None = output_script_path,
    parameters: list[dict] | None = template_param,
    classname: str = Option("Model", "--classname", "-n", help="Name of the generated layer class."),
    structured_output: bool = Option(
        True,
        help="Return the layer outputs as a dict keyed by output name instead of positionally. Defaults to true, "
        "which overrides the configuration's STRUCTURED_OUTPUT; pass --no-structured-output for positional output. "
        "Ignored with --sublayer: the selected layer's own configuration decides.",
    ),
    sublayer: str | None = Option(
        None, "--sublayer", help="The reference to a sublayer in the template to build instead of the root layer."
    ),
) -> None:
    """Create a Keras layer class (a keras.layers.Layer subclass) from the given configuration file and parameters."""
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
    training_mode: bool = training_mode,
    warmup_runs: int = warmup_runs,
    times: int = times,
    batch_size: int = batch_size,
) -> None:
    """Measure the average inference time of a Keras model."""
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
