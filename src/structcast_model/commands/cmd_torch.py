"""PyTorch related commands for the StructCast Model CLI application."""

from typing import TYPE_CHECKING, Any, Literal

from structcast.utils.base import load_yaml_from_string
from structcast.utils.security import configure_security
from typer import Argument, Option, Typer

from structcast_model.base_trainer import BaseInfo
from structcast_model.commands.utils import bool_or_dict_parser, dict_parser, reduce_dict, tensor_shape_parser

if TYPE_CHECKING:
    import calflops
    import mlflow
    import numpy as np
    import ptflops
    from structcast.core import instantiator
    import tqdm

    from structcast_model.builders import torch_builder
    from structcast_model.torch import trainer as torch_trainer
    import torch
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    calflops = LazyModuleImporter("calflops")
    mlflow = LazyModuleImporter("mlflow")
    np = LazyModuleImporter("numpy")
    ptflops = LazyModuleImporter("ptflops")
    instantiator = LazyModuleImporter("structcast.core.instantiator")
    tqdm = LazyModuleImporter("tqdm")
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


def _init_models(patterns: list[dict[str, Any]], shapes: dict[str, Any] | None) -> list[tuple[str, "torch.nn.Module"]]:
    models = []
    inputs = None if shapes is None else torch_trainer.create_torch_inputs(shapes)
    for raw in patterns:
        if len(raw) != 1:
            raise ValueError(f"Each model pattern should contain exactly one model definition. Got: {raw}")
        model_name, ptn = list(raw.items())[0]
        model: torch.nn.Module = _instantiate(ptn)
        if inputs is not None:
            model(**inputs)  # forward first to make sure the model is built
        models.append((model_name, model))
    return models


@app.command()
def train(
    model_patterns: list[dict] = Argument(
        parser=dict_parser,
        help="The object patterns used to instantiate models. "
        "For example, if the model is defined as `model_name = my_package.MyModel(...)`, then the pattern should be "
        '"model_name: [_obj_, {_addr_: my_package.MyModel, _file_: my_package.py}, {_call_: {...}}]" or '
        '"model_name: [_obj_, [_addr_, my_package.MyModel, my_package.py], {_call_: {...}}]".',
    ),
    shapes: list[dict] | None = shapes,
    loss_pattern: dict = Option(
        ...,
        "--loss",
        "-l",
        parser=dict_parser,
        help="The object pattern used to instantiate the loss module. "
        "For example, if the loss module is defined as `my_package.MyLoss(...)`, then the pattern should be "
        '"[_obj_, {_addr_: my_package.MyLoss, _file_: my_package.py}, {_call_: {...}}]" or '
        '"[_obj_, [_addr_, my_package.MyLoss, my_package.py], {_call_: {...}}]".',
    ),
    metric_pattern: dict | None = Option(
        None,
        "--metric",
        "-m",
        parser=dict_parser,
        help="The object pattern used to instantiate the metric module. "
        "For example, if the metric module is defined as `my_package.MyMetric(...)`, then the pattern should be "
        '"[_obj_, {_addr_: my_package.MyMetric, _file_: my_package.py}, {_call_: {...}}]" or '
        '"[_obj_, [_addr_, my_package.MyMetric, my_package.py], {_call_: {...}}]".',
    ),
    backward_pattern: dict = Option(
        ...,
        "--backward",
        "-b",
        parser=dict_parser,
        help="The object pattern used to instantiate the backward class. "
        "For example, if the backward class is defined as `my_package.MyBackward(...)`, then the pattern should be "
        '"[_obj_, {_addr_: my_package.MyBackward, _file_: my_package.py}, {_call_: {...}}]" or '
        '"[_obj_, [_addr_, my_package.MyBackward, my_package.py], {_call_: {...}}]".',
    ),
    compile_pattern: dict[str, Any] | None = Option(
        None,
        "--compile",
        "-c",
        parser=bool_or_dict_parser,
        help='Whether to compile the model using "torch.compile". '
        'Can be set to true/false or a dictionary of keyword arguments for "torch.compile".',
    ),
    epochs: int = Option(1, "--epochs", "-e", help="Number of training epochs."),
    start_epoch: int = Option(1, help="Starting epoch number."),
    device: str | None = device,
    seed: int = Option(0, help="Random seed for reproducibility."),
    matmul_precision: Literal["highest", "high", "medium"] = Option("high", help="Matrix multiplication precision."),
    experiment: str = Option("experiment", "--experiment", "-x", help="Experiment name for MLflow logging."),
    ci: bool = Option(
        False,
        help="Whether to run in CI mode. "
        "If true, it will print the criteria at the end of each epoch instead of using a progress bar.",
    ),
) -> None:
    if not model_patterns:
        raise ValueError("At least one model pattern must be provided.")
    configure_security(allowed_modules_check=False)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision(matmul_precision)
    torch.manual_seed(seed)
    np.random.seed(seed)
    input_shapes = reduce_dict(shapes)
    compile_kw = instantiator.instantiate(compile_pattern)
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        with torch.device(torch_trainer.get_torch_device(device)):
            models = _init_models(model_patterns, input_shapes)
            loss: torch.nn.Module = _instantiate(loss_pattern)
            metric: torch.nn.Module | None = _instantiate(metric_pattern) if metric_pattern else None
            if compile_kw is not None:
                models = [(n, torch.compile(m, **compile_kw)) for n, m in models]
                loss = torch.compile(loss, **compile_kw)
                if metric is not None:
                    metric = torch.compile(metric, **compile_kw)
            backward = _instantiate(backward_pattern)(**dict(models))
            trainer = torch_trainer.TorchTrainer()

            def _log_criteria(info: BaseInfo) -> str:
                """Format the criteria."""
                lrs, logs = backward.learning_rates, info.logs()
                values = "\n\t".join([f"{k}: {v:.4e}" for k, v in {**logs, **lrs}.items()])
                mlflow.log_metrics({**logs, **lrs}, step=info.epoch)
                return f"Epoch {info.epoch}:\n\t{values}"

            if ci:
                trainer.on_epoch_end.append(lambda i: print(_log_criteria(i)))
            else:
                pbar = tqdm.tqdm(unit="batch")


__all__ = ["app"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
