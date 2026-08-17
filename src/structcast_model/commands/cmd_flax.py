"""Flax related commands for the StructCast Model CLI application."""

from time import time
from typing import TYPE_CHECKING, Any

from typer import Argument, Option, Typer

from structcast_model.commands.utils import (
    TEMPLATE_PARAM_HELP,
    bool_or_path_or_dict_parser,
    dict_parser,
    instantiate_object,
    object_pattern_help,
    path_or_any_parser,
    reduce_dict,
    tensor_shape_parser,
)

if TYPE_CHECKING:
    import jax
    from structcast.core import instantiator

    from flax import nnx
    from structcast_model.builders import flax_builder
    from structcast_model.flax import trainer as flax_trainer
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    jax = LazyModuleImporter("jax")
    instantiator = LazyModuleImporter("structcast.core.instantiator")
    nnx = LazyModuleImporter("flax.nnx")
    flax_builder = LazyModuleImporter("structcast_model.builders.flax_builder")
    flax_trainer = LazyModuleImporter("structcast_model.flax.trainer")


app = Typer(no_args_is_help=True)
creator = Typer(no_args_is_help=True)
app.add_typer(creator, name="create", help="Commands for creating Flax nnx modules.")

template_param = Option(
    None,
    "--parameter",
    "-p",
    parser=dict_parser,
    help=TEMPLATE_PARAM_HELP + ' For example: --parameter "model: {input_size: 128, output_size: 10}"',
)
output_script_path = Option(
    None,
    "--output",
    "-o",
    help='Path of the generated Python script. Defaults to the snake_case --classname plus ".py" in the current '
    "directory. An existing file is overwritten and missing parent directories are created.",
)
model_pattern = Argument(
    parser=path_or_any_parser,
    help=object_pattern_help("the model", "MyModel")
    + " The pattern may be given inline or as a path to a YAML/JSON file holding it.",
)
shapes = Option(
    None,
    "--shape",
    "-s",
    parser=tensor_shape_parser,
    help="Input tensor shapes for one sample, without the batch dimension, as a YAML mapping of input name to "
    "specification. Compact form: 'image: [224, 224, 3]'. Explicit form: "
    "'tokens: {_SHAPE_: [512], _DTYPE_: int64, _INIT_: jax.numpy.zeros}'. Specifications may nest in mappings and "
    "lists. _DTYPE_ defaults to bfloat16 (not float32), and an integer dtype without _INIT_ falls back to zeros "
    "with a warning. Omit it only when the model declares INPUT_SHAPES itself.",
)
device = Option(
    None,
    "--device",
    "-d",
    help='Device the dummy inputs are placed on, named "<platform>:<index>" as listed by JAX (e.g. "cpu:0", '
    '"gpu:0"). If not specified, the first available JAX device is used.',
)
compile_pattern: dict[str, Any] | None = Option(
    None,
    "--compile",
    "-c",
    parser=bool_or_path_or_dict_parser,
    help='Whether to compile the model with "nnx.jit". Accepts true/false, a path to an existing YAML/JSON file, or '
    'a mapping of keyword arguments for "nnx.jit". Omitted or false means the model is not compiled; true compiles '
    "with defaults.",
)


@creator.command(name="model")
def create_model(
    cfg_path: str = Argument(..., help="Path to the model configuration file."),
    output: str | None = output_script_path,
    parameters: list[dict] | None = template_param,
    classname: str = Option("Model", "--classname", "-c", help="Name of the generated model class."),
    structured_output: bool = Option(
        True,
        help="Return the module outputs as a dict keyed by output name instead of positionally. Defaults to true, "
        "which overrides the configuration's STRUCTURED_OUTPUT; pass --no-structured-output to force plain output. "
        "Ignored with --sublayer: the selected layer's own configuration decides.",
    ),
    sublayer: str | None = Option(
        None, "--sublayer", "-s", help="The reference to a sublayer in the template to build instead of the root layer."
    ),
) -> None:
    """Create a Flax nnx module from the given configuration file and parameters."""
    flax_builder.FlaxBuilder.from_path(cfg_path)(
        parameters=reduce_dict(parameters),
        classname=classname,
        forced_structured_output=structured_output,
        user_defined_layer=sublayer,
    )(output)


@app.command(name="time")
def measure_inference_time(
    model_pattern: Any = model_pattern,
    shapes: dict | None = shapes,
    device: str | None = device,
    compile_pattern: dict[str, Any] | None = compile_pattern,
    training_mode: bool = Option(
        False,
        help="Whether to select the model's training-mode view (training / deterministic / use_running_average) for "
        "the measurement, which can change the timing through dropout and batch norm. Flags the model does not "
        "declare are ignored, and --training-mode-kwargs replaces these flags entirely when given.",
    ),
    training_mode_kwargs_pattern: dict[str, Any] | None = Option(
        None,
        "--training-mode-kwargs",
        parser=bool_or_path_or_dict_parser,
        help='Keyword arguments for "nnx.view", e.g. --training-mode-kwargs "{training: true, deterministic: false, '
        'use_running_average: false}". Can also be a path to a YAML/JSON file holding them; "false" is the same as '
        'omitting the option, and "true" applies no view flags at all. When given, the mapping applies whether or '
        "not --training-mode is set, replacing the view flags derived from it. If not specified, the view is "
        "derived from --training-mode as {training: <flag>, deterministic: <not flag>, "
        "use_running_average: <not flag>}.",
    ),
    warmup_runs: int = Option(2, "--warmup-runs", "-w", help="Number of warmup runs before measuring inference time."),
    times: int = Option(10, "--times", "-t", help="Number of iterations to measure the inference time."),
    batch_size: int = Option(
        1, "--batch-size", "-b", help="Batch size for the input tensors during inference time measurement."
    ),
) -> None:
    """Measure the average inference time of a Flax model."""
    jax_device = flax_trainer.get_jax_device(device)
    training_mode_kw = (
        {"training": training_mode, "deterministic": not training_mode, "use_running_average": not training_mode}
        if training_mode_kwargs_pattern is None
        else instantiator.instantiate(training_mode_kwargs_pattern)
    )
    print("Initializing the model...")
    model = instantiate_object(model_pattern)
    shapes = flax_trainer.resolve_input_shapes(model, shapes)
    model = nnx.view(model, raise_if_not_found=False, **training_mode_kw)
    if compile_pattern is None:
        print("Skipping compilation...")
    else:
        print("Compiling the model...")
        model = nnx.jit(model, **instantiator.instantiate(compile_pattern))

    def _measure_single_run() -> float:
        inputs = jax.device_put(flax_trainer.create_jax_inputs(shapes, batch_size=batch_size), device=jax_device)
        start_time = time()
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), model(**inputs))
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
