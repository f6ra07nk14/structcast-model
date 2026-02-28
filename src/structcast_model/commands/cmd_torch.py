"""PyTorch related commands for the StructCast Model CLI application."""

from typing import TYPE_CHECKING, Any, Literal

from structcast.utils.base import load_yaml_from_string
from structcast.utils.security import configure_security
from typer import Argument, Option, Typer

from structcast_model.commands.utils import dict_parser, reduce_dict, tensor_shape_parser

if TYPE_CHECKING:
    import calflops
    import ptflops
    from structcast.core import instantiator

    from structcast_model.builders import torch_builder
    from structcast_model.torch import trainer as torch_trainer
    import torch
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    calflops = LazyModuleImporter("calflops")
    ptflops = LazyModuleImporter("ptflops")
    instantiator = LazyModuleImporter("structcast.core.instantiator")
    torch_builder = LazyModuleImporter("structcast_model.builders.torch_builder")
    torch_trainer = LazyModuleImporter("structcast_model.torch.trainer")
    torch = LazyModuleImporter("torch")


app = Typer(no_args_is_help=True)
creator = Typer(no_args_is_help=True)
app.add_typer(creator, name="create", help="Commands for creating PyTorch models and backward classes.")

template_param = Option(
    None,
    "--parameter",
    "-p",
    parser=dict_parser,
    help="Parameters to format the template configuration file with. Can be specified multiple times. "
    'Each parameter should be in the format of "key: {...}", where `key` is the name of the parameter group, '
    "and the value is a dictionary of keyword arguments for formatting the template. "
    'For example: --parameter "model: {input_size: 128, output_size: 10}" --parameter "optimizer: {lr: 0.001}"',
)
output_script_path = Option(None, "--output", "-o", help="Output script path (Python).")
model_pattern = Argument(
    parser=load_yaml_from_string,
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
    help="Input tensor shapes as a dictionary, e.g., 'image: [3, 224, 224]'.",
)
device = Option(
    None,
    "--device",
    "-d",
    help='Computation device to use, either "cpu" or "cuda". '
    'If not specified, it will use "cuda" if available, otherwise "cpu".',
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
    """Create a PyTorch model from the given configuration file and parameters."""
    torch_builder.TorchBuilder.from_path(cfg_path)(
        parameters=reduce_dict(parameters),
        classname=classname,
        forced_structured_output=structured_output,
        user_defined_layer=sublayer,
    )(output)


@creator.command(name="backward")
def create_backward(
    cfg_path: str = Argument(..., help="Path to the backward configuration file."),
    output: str | None = output_script_path,
    parameters: list[dict] | None = template_param,
    classname: str = Option("Backward", "--classname", "-c", help="Name the backward class."),
) -> None:
    """Create a PyTorch backward class from the given configuration file and parameters."""
    builder = torch_builder.TorchBackwardBuilder.from_path(cfg_path)
    builder(parameters=reduce_dict(parameters), classname=classname)(output)


def _instantiate(raw: Any) -> Any:
    return instantiator.ObjectPattern.model_validate(raw).build().runs[0]


@app.command(name="ptflops")
def call_ptflops(
    model_pattern: Any = model_pattern,
    shapes: list[dict] | None = shapes,
    output_precision: int = Option(4, help="Decimal precision for FLOPs and parameters output."),
    flops_units: Literal["GMac", "MMac", "KMac"] = Option("GMac", help="Units for FLOPs: GMac, MMac, or KMac."),
    param_units: Literal["M", "K", "B"] = Option(
        "M", help="Units for parameters: M (millions), K (thousands), or B (billions)."
    ),
    backend: Literal["pytorch", "aten"] = Option(
        "aten", help='Backend for FLOPs computation. Note: Don\'t use "pytorch" backend for transformer architectures.'
    ),
    device: str | None = device,
) -> None:
    """Calculate the FLOPs and number of parameters of a PyTorch model using ptflops."""
    configure_security(allowed_modules_check=False)
    with torch.device(torch_trainer.get_torch_device(device)):
        inputs = torch_trainer.create_torch_inputs(reduce_dict(shapes))
        model: torch.nn.Module = _instantiate(model_pattern)
        model(**inputs)  # forward first to make sure the model is built
        flops, params = ptflops.get_model_complexity_info(
            model=model,
            input_res=(1,),
            print_per_layer_stat=True,
            input_constructor=lambda _: inputs,
            verbose=True,
            ignore_modules=[],
            custom_modules_hooks={},
            backend=backend,
            output_precision=output_precision,
            flops_units=flops_units,
            param_units=param_units,
        )
    if flops:
        print(f"{'Computational complexity: ':<30}  {flops:<8}")
    if params:
        print(f"{'Number of parameters: ':<30}  {params:<8}")


@app.command(name="calflops")
def call_calflops(
    model_pattern: Any = model_pattern,
    shapes: list[dict] | None = shapes,
    include_bp: bool = Option(False, help="Whether to include backpropagation in FLOPs computation."),
    output_precision: int = Option(4, help="Decimal precision for FLOPs and parameters output."),
    bp_factor: float = Option(2.0, help="Factor to multiply the forward FLOPs by to estimate backpropagation FLOPs."),
    device: str | None = device,
) -> None:
    """Calculate the FLOPs and number of parameters of a PyTorch model using calflops."""
    configure_security(allowed_modules_check=False)
    with torch.device(torch_trainer.get_torch_device(device)):
        inputs = torch_trainer.create_torch_inputs(reduce_dict(shapes))
        model: torch.nn.Module = _instantiate(model_pattern)
        model(**inputs)  # forward first to make sure the model is built
        flops, macs, params = calflops.calculate_flops(
            model=model,
            input_shape=None,
            args=[],
            kwargs=inputs,
            forward_mode="forward",
            include_backPropagation=include_bp,
            compute_bp_factor=bp_factor,
            print_results=True,
            print_detailed=True,
            output_as_string=True,
            output_precision=output_precision,
            output_unit=None,
            ignore_modules=None,
        )
    print(f"FLOPs: {flops}")
    print(f"MACs: {macs}")
    print(f"Parameters: {params}")


__all__ = ["app"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
