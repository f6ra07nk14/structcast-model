"""Flax related commands for the StructCast Model CLI application."""

from time import time
from typing import TYPE_CHECKING, Any

from structcast.utils.security import configure_security
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
)
compile_pattern: dict[str, Any] | None = Option(
    None,
    "--compile",
    "-c",
    parser=bool_or_path_or_dict_parser,
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
        help="Whether to set the model to training mode during inference time measurement. "
        "This can affect the inference time due to differences in behavior (e.g., dropout, batch norm).",
    ),
    training_mode_kwargs_pattern: dict[str, Any] | None = Option(
        None,
        "--training-mode-kwargs",
        parser=bool_or_path_or_dict_parser,
        help="Additional keyword arguments for the model view when `--training-mode` is set. "
        "This should be a dictionary of keyword arguments for `nnx.view` when `--training-mode` is true. "
        'For example: --training-mode-kwargs "{deterministic: false, use_running_average: false}"',
    ),
    warnup_runs: int = Option(2, "--warmup-runs", "-w", help="Number of warmup runs before measuring inference time."),
    times: int = Option(10, "--times", "-t", help="Number of iterations to measure the inference time."),
    batch_size: int = Option(
        1, "--batch-size", "-b", help="Batch size for the input tensors during inference time measurement."
    ),
) -> None:
    """Measure the average inference time of a Flax model."""
    configure_security(allowed_modules_check=False)
    jax_device = flax_trainer.get_jax_device(device)
    training_mode_kw = (
        {"training": training_mode, "deterministic": not training_mode, "use_running_average": not training_mode}
        if training_mode_kwargs_pattern is None
        else instantiate_object(training_mode_kwargs_pattern)
    )
    print("Initializing the model...")
    model = instantiate_object(model_pattern)
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

    print(f"Running {warnup_runs} warmup runs...")
    for _ in range(warnup_runs):
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
