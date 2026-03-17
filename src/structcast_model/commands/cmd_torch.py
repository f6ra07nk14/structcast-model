"""PyTorch related commands for the StructCast Model CLI application."""

from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from structcast.utils.base import dump_yaml_to_string
from structcast.utils.security import configure_security
from typer import Argument, Option, Typer

from structcast_model.base_trainer import BaseInfo, callbacks_session, get_dataset_size
from structcast_model.commands.utils import (
    bool_or_path_or_dict_parser,
    dict_parser,
    path_or_any_parser,
    reduce_dict,
    tensor_shape_parser,
)

if TYPE_CHECKING:
    import calflops
    import mlflow
    import numpy as np
    import ptflops
    from structcast.core import instantiator
    import timm
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
    timm = LazyModuleImporter("timm")
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
    help="Parameters to format the template configuration file with. "
    'Each parameter should be in the format of "key: {...}", where `key` is the name of the parameter group, '
    "and the value is a dictionary of keyword arguments for formatting the template. "
    'For example: --parameter "model: {input_size: 128, output_size: 10}" --parameter "optimizer: {lr: 0.001}"',
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


def _compile_module(module: Any, compile_kw: dict[str, Any] | None) -> Any:
    """Compile a PyTorch module if compile_kw is provided."""
    return module if compile_kw is None else torch.compile(module, **compile_kw)


def _instantiate_models(patterns: list[dict]) -> "OrderedDict[str, Any]":
    """Instantiate models from a list of name-pattern mappings."""
    res: OrderedDict[str, Any] = OrderedDict()
    for raw in patterns:
        if len(raw) != 1:
            raise ValueError(f"Each model pattern should contain exactly one model definition. Got: {raw}")
        model_name, ptn = list(raw.items())[0]
        res[model_name] = _instantiate(ptn)
    return res


def _get_module_outputs(module: Any, default: list[str] | None, name: str) -> list[str]:
    """Return output names from a module attribute or the provided default, raising if neither is available."""
    if hasattr(module, "outputs"):
        return module.outputs
    if default:
        return default
    raise ValueError(
        f'Module "{name}" does not have an "outputs" attribute. '
        f'Please provide default outputs using the "--{name}-outputs" option.'
    )


def _get_state_dict(**kwargs: Any) -> dict[str, Any]:
    """Return a mapping of name to state dict for all given modules."""
    return {n: m.state_dict() for n, m in kwargs.items()}


def _on_best(info: BaseInfo, target: str, value: float, save: bool, **kwargs: Any) -> None:
    """Log best metric value and optionally save model state dict to MLflow."""
    mlflow.log_metric(f"{target}_best", value, step=info.epoch)
    if save:
        mlflow.pytorch.log_state_dict(_get_state_dict(**kwargs), f"{target}_best")


def _save_training_state(info: BaseInfo, **kwargs: Any) -> None:
    """Save full training state (models, optimizers, grad scalers, EMA, meta) to MLflow."""
    backward = cast("torch_trainer.TorchTrainer", info).backward
    wrapper = cast("torch_trainer.TorchTrainer", info).inference_wrapper
    states: dict[str, Any] = {
        "models": _get_state_dict(**kwargs),
        "optimizers": _get_state_dict(**getattr(backward, "optimizers", {})),
        "grad_scalers": _get_state_dict(**getattr(backward, "grad_scalers", {})),
        "meta": {"epoch": info.epoch, "step": info.step, "update": info.update},
    }
    if wrapper is not None:
        states["ema"] = _get_state_dict(**cast("torch_trainer.TimmEmaWrapper", wrapper).models)
    mlflow.pytorch.log_state_dict(states, artifact_path="training_state")


def _log_criteria(info: BaseInfo) -> str:
    """Format current epoch criteria as YAML and log metrics to MLflow, returning a display string."""
    values = {**getattr(cast("torch_trainer.TorchTrainer", info).backward, "learning_rates", {}), **info.logs()}
    mlflow.log_metrics(values, step=info.epoch)
    return f"epoch: {info.epoch}\n{dump_yaml_to_string(values)}"


@app.command(name="ptflops")
def call_ptflops(
    model_pattern: Any = model_pattern,
    shapes: dict | None = shapes,
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
        model = _instantiate(model_pattern)
        inputs, _ = torch_trainer.initial_model(model, shapes)
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
    shapes: dict | None = shapes,
    include_bp: bool = Option(False, help="Whether to include backpropagation in FLOPs computation."),
    output_precision: int = Option(4, help="Decimal precision for FLOPs and parameters output."),
    bp_factor: float = Option(2.0, help="Factor to multiply the forward FLOPs by to estimate backpropagation FLOPs."),
    device: str | None = device,
) -> None:
    """Calculate the FLOPs and number of parameters of a PyTorch model using calflops."""
    configure_security(allowed_modules_check=False)
    with torch.device(torch_trainer.get_torch_device(device)):
        model = _instantiate(model_pattern)
        inputs, _ = torch_trainer.initial_model(model, shapes)
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


@app.command()
@callbacks_session()
def train(  # noqa: PLR0913,PLR0915
    model_patterns: list[dict] = Argument(
        parser=dict_parser,
        help="The object patterns used to instantiate models. "
        "For example, if the model is defined as `model_name = my_package.MyModel(...)`, then the pattern should be "
        '"model_name: [_obj_, {_addr_: my_package.MyModel, _file_: my_package.py}, {_call_: {...}}]" or '
        '"model_name: [_obj_, [_addr_, my_package.MyModel, my_package.py], {_call_: {...}}]".',
    ),
    shapes: list[dict] | None = shapes,
    device: str | None = device,
    ema: dict[str, Any] | None = Option(
        None,
        parser=bool_or_path_or_dict_parser,
        help="Whether to use EMA (Exponential Moving Average) for the model during training. "
        "Can be set to true/false, a path to a YAML file, or a dictionary of keyword arguments "
        "for the EMA wrapper (e.g., decay rate).",
    ),
    ema_device: str | None = Option(
        None, help="Device for the EMA model. If not specified, it will use the same device as the main model."
    ),
    loss_pattern: Any = Option(
        ...,
        "--loss",
        "-l",
        parser=path_or_any_parser,
        help="The object pattern used to instantiate the loss module. "
        "For example, if the loss module is defined as `my_package.MyLoss(...)`, then the pattern should be "
        '"[_obj_, {_addr_: my_package.MyLoss, _file_: my_package.py}, {_call_: {...}}]" or '
        '"[_obj_, [_addr_, my_package.MyLoss, my_package.py], {_call_: {...}}]".',
    ),
    loss_outputs: list[str] | None = Option(
        None,
        "--loss-outputs",
        "-LO",
        help="Default outputs for the loss module if it doesn't have an 'outputs' attribute.",
    ),
    metric_pattern: Any | None = Option(
        None,
        "--metric",
        "-m",
        parser=path_or_any_parser,
        help="The object pattern used to instantiate the metric module. "
        "For example, if the metric module is defined as `my_package.MyMetric(...)`, then the pattern should be "
        '"[_obj_, {_addr_: my_package.MyMetric, _file_: my_package.py}, {_call_: {...}}]" or '
        '"[_obj_, [_addr_, my_package.MyMetric, my_package.py], {_call_: {...}}]".',
    ),
    metric_outputs: list[str] | None = Option(
        None,
        "--metric-outputs",
        "-MO",
        help="Default outputs for the metric module if it doesn't have an 'outputs' attribute.",
    ),
    backward_pattern: Any = Option(
        ...,
        "--backward",
        "-b",
        parser=path_or_any_parser,
        help="The object pattern used to instantiate the backward class. "
        "For example, if the backward class is defined as `my_package.MyBackward(...)`, then the pattern should be "
        '"[_obj_, {_addr_: my_package.MyBackward, _file_: my_package.py}, {_call_: {...}}]" or '
        '"[_obj_, [_addr_, my_package.MyBackward, my_package.py], {_call_: {...}}]".',
    ),
    mixed_precision_type: Literal["bfloat16", "float16"] | None = Option(
        None,
        help="Default mixed precision type to use during training when mixed precision is enabled. "
        "This can be overridden by the backward class if it has its own mixed precision type specified.",
    ),
    compile_pattern: dict[str, Any] | None = Option(
        None,
        "--compile",
        "-c",
        parser=bool_or_path_or_dict_parser,
        help='Whether to compile the model using "torch.compile". '
        'Can be set to true/false, a path to a YAML file, or a dictionary of keyword arguments for "torch.compile".',
    ),
    epochs: int = Option(1, "--epochs", "-e", help="Number of training epochs."),
    start_epoch: int = Option(1, help="Starting epoch number."),
    training_dataset_pattern: Any = Option(
        ...,
        "--training-dataset",
        "-t",
        parser=path_or_any_parser,
        help="The object pattern used to instantiate the training dataset. "
        "For example, if the dataset is defined as `my_package.MyDataset(...)`, then the pattern should be "
        '"[_obj_, {_addr_: my_package.MyDataset, _file_: my_package.py}, {_call_: {...}}]" or '
        '"[_obj_, [_addr_, my_package.MyDataset, my_package.py], {_call_: {...}}]".',
    ),
    validation_dataset_pattern: Any | None = Option(
        None,
        "--validation-dataset",
        "-v",
        parser=path_or_any_parser,
        help="The object pattern used to instantiate the validation dataset. "
        "For example, if the dataset is defined as `my_package.MyDataset(...)`, then the pattern should be "
        '"[_obj_, {_addr_: my_package.MyDataset, _file_: my_package.py}, {_call_: {...}}]" or '
        '"[_obj_, [_addr_, my_package.MyDataset, my_package.py], {_call_: {...}}]".',
    ),
    validation_frequency: int = Option(1, "--validation-frequency", "-f", help="Frequency of validation (in epochs)."),
    lower_criteria: list[str] = Option(
        ...,
        "--lower-criterion",
        "-LC",
        default_factory=list,
        help="Criterion names that require lower values.",
    ),
    higher_criteria: list[str] = Option(
        ...,
        "--higher-criterion",
        "-HC",
        default_factory=list,
        help="Criterion names that require higher values.",
    ),
    save_criteria: list[str] = Option(
        ...,
        "--save-criterion",
        "-SC",
        default_factory=list,
        help="Criterion names to monitor for saving the best model. "
        "Should be a subset of lower_criteria and higher_criteria.",
    ),
    seed: int = Option(42, envvar="SEED", help="Random seed for reproducibility."),
    matmul_precision: Literal["highest", "high", "medium"] = Option(
        "high", envvar="MATMUL_PRECISION", help="Matrix multiplication precision."
    ),
    experiment: str = Option(
        "experiment", "--experiment", "-E", envvar="EXPERIMENT", help="Experiment name for MLflow logging."
    ),
    log_arguments: list[dict] | None = Option(
        None, "--log-arguments", "-K", parser=dict_parser, help="Additional arguments to log in MLflow."
    ),
    log_artifacts: list[Path] | None = Option(None, "--log-artifacts", "-A", help="Artifacts to log in MLflow."),
    ci: bool = Option(
        False,
        help="Whether to run in CI mode. "
        "If true, it will print the criteria at the end of each epoch instead of using a progress bar.",
    ),
    dist_backend: str | None = Option(
        None,
        envvar="DIST_BACKEND",
        help="Distributed backend to use (e.g., 'nccl', 'gloo'). If None, it will be automatically selected.",
    ),
    dist_url: str | None = Option(
        None, envvar="DIST_URL", help="URL to use for setting up distributed training. If None, it will use 'env://'."
    ),
) -> None:
    """Train a PyTorch model with MLflow tracking."""
    if not model_patterns:
        raise ValueError("At least one model pattern must be provided.")
    configure_security(allowed_modules_check=False)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision(matmul_precision)
    torch.manual_seed(seed)
    np.random.seed(seed)
    input_shapes = reduce_dict(shapes)
    device, global_rank, _, world_size, distributed = torch_trainer.initial_distributed_env(
        device=device, dist_backend=dist_backend, dist_url=dist_url, return_dict=False
    )
    is_main = global_rank == 0
    compile_fn = partial(_compile_module, compile_kw=instantiator.instantiate(compile_pattern))
    dist_fn = partial(torch.nn.parallel.DistributedDataParallel, device_ids=[device]) if distributed else lambda m: m
    training_dataset = _instantiate(training_dataset_pattern)
    validation_dataset = _instantiate(validation_dataset_pattern) if validation_dataset_pattern else None
    if is_main:
        print("Count the dataset sizes...")
    steps_per_epoch = get_dataset_size(training_dataset)
    validation_steps = 0 if validation_dataset is None else get_dataset_size(validation_dataset)
    if is_main:
        print(f"Training dataset size: {steps_per_epoch} steps.")
        print(f"Validation dataset size: {validation_steps} steps.")
    with torch.device(device):
        models = _instantiate_models(model_patterns)
        torch_trainer.initial_model(_instantiate_models(model_patterns), input_shapes)
        loss = compile_fn(_instantiate(loss_pattern))
        metric = compile_fn(_instantiate(metric_pattern)) if metric_pattern else None
        loss_outputs = _get_module_outputs(loss, loss_outputs, "loss")
        metric_outputs = None if metric is None else _get_module_outputs(metric, metric_outputs, "metric")
        tracker = torch_trainer.TorchTracker.from_criteria(loss_outputs, metric_outputs, compile_fn)
        backward = _instantiate(backward_pattern)(**models)
    mixed_precision_type = getattr(backward, "mixed_precision_type", mixed_precision_type)
    autocast = torch_trainer.get_autocast(mixed_precision_type, device)
    inference_wrapper = None
    if ema is not None:
        inference_wrapper = torch_trainer.TimmEmaWrapper.from_models(
            models,
            compile_fn=compile_fn,
            device=None if ema_device is None else torch.device(ema_device),
            **instantiator.instantiate(ema),
        )
    models = OrderedDict((n, compile_fn(dist_fn(m))) for n, m in models.items())
    step_kw = {"models": list(models), "losses": loss, "metrics": metric, "autocast": autocast}
    trainer = torch_trainer.TorchTrainer(
        device=device,
        inference_wrapper=inference_wrapper,
        training_step=torch_trainer.TrainingStep(**step_kw),
        validation_step=torch_trainer.ValidationStep(**step_kw),
        backward=backward,
        tracker=tracker,
    )
    if is_main:
        if ci:
            trainer.on_epoch_end.register("log_criteria", lambda i, **_: print(_log_criteria(i)))  # type: ignore[arg-type]
        else:
            pbar = tqdm.tqdm(unit="batch")

            def _update_criteria(info: BaseInfo, criteria: list[str], **_: Any) -> None:
                logs = info.logs()
                pbar.update()
                pbar.set_postfix([(n, logs[n]) for n in criteria])

            trainer.on_training_begin.register("pbar_reset_training", lambda i, **_: pbar.reset(steps_per_epoch))  # type: ignore[arg-type]
            train_losses = [f"{trainer.training_prefix}{n}" for n in loss_outputs]
            pbar_update_training = partial(_update_criteria, criteria=train_losses)
            trainer.on_training_step_end.register("pbar_update_training", pbar_update_training)
            trainer.on_training_end.register("pbar_refresh_training", lambda i, **_: pbar.refresh())  # type: ignore[arg-type]
            trainer.on_validation_begin.register("pbar_reset_validation", lambda i, **_: pbar.reset(validation_steps))  # type: ignore[arg-type]
            valid_losses = [f"{trainer.validation_prefix}{n}" for n in loss_outputs]
            pbar_update_validation = partial(_update_criteria, criteria=valid_losses)
            trainer.on_validation_step_end.register("pbar_update_validation", pbar_update_validation)
            trainer.on_validation_end.register("pbar_refresh_validation", lambda i, **_: pbar.refresh())  # type: ignore[arg-type]
            trainer.on_epoch_end.register("pbar_log_criteria", lambda i, **_: pbar.write(_log_criteria(i)))  # type: ignore[arg-type]
        trainer.on_epoch_end.register("save_training_state", _save_training_state)
        for criterion_name in higher_criteria:
            on_best = partial(_on_best, save=criterion_name in save_criteria)
            trainer.on_epoch_end.register(
                f"best_{criterion_name}",
                torch_trainer.TorchBestCriterion(target=criterion_name, mode="max", on_best=[on_best]),
            )
        for criterion_name in lower_criteria:
            on_best = partial(_on_best, save=criterion_name in save_criteria)
            trainer.on_epoch_end.register(
                f"best_{criterion_name}",
                torch_trainer.TorchBestCriterion(target=criterion_name, mode="min", on_best=[on_best]),
            )
    fit_kwargs: dict[str, Any] = {
        "epochs": epochs,
        "training_dataset": training_dataset,
        "validation_dataset": validation_dataset,
        "start_epoch": start_epoch,
        "validation_frequency": validation_frequency,
    }
    try:
        if is_main:
            arguments = {
                **reduce_dict(log_arguments),
                "models": model_patterns,
                "parameters": {n: sum(p.numel() for p in m.parameters() if p.requires_grad) for n, m in models.items()},
                "shapes": input_shapes,
                "device": device,
                "distributed": distributed,
                "world_size": world_size,
                "ema": ema,
                "ema_device": ema_device,
                "loss": loss_pattern,
                "loss_outputs": loss_outputs,
                "metric": metric_pattern,
                "metric_outputs": metric_outputs,
                "backward": backward_pattern,
                "mixed_precision_type": mixed_precision_type,
                "compile": compile_pattern,
                "epochs": epochs,
                "start_epoch": start_epoch,
                "training_dataset": training_dataset_pattern,
                "validation_dataset": validation_dataset_pattern,
                "validation_frequency": validation_frequency,
                "lower_criteria": lower_criteria,
                "higher_criteria": higher_criteria,
                "save_criteria": save_criteria,
                "seed": seed,
                "matmul_precision": matmul_precision,
                "experiment": experiment,
                "ci": ci,
            }
            mlflow.set_experiment(experiment)
            with mlflow.start_run():
                mlflow.log_param("cuda_version", torch.version.cuda)
                mlflow.log_param("torch_version", torch.__version__)
                mlflow.log_param("timm_version", timm.__version__)
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("steps_per_epoch", steps_per_epoch)
                mlflow.log_param("validation_steps", validation_steps)
                if hasattr(backward, "param_group_names"):
                    mlflow.log_dict(backward.param_group_names, "param_groups.yaml")
                mlflow.log_dict(arguments, "arguments.yaml")
                for artifact in log_artifacts or []:
                    mlflow.log_artifact(str(artifact))
                print(f"Registered callbacks:\n{dump_yaml_to_string(trainer.describe())}")
                try:
                    trainer.fit(**fit_kwargs, **models)
                except KeyboardInterrupt:
                    print("Training interrupted by user. Saving current state to MLflow.")
                    _save_training_state(trainer, **models)
        else:
            trainer.fit(**fit_kwargs, **models)
    finally:
        if distributed:
            torch.distributed.destroy_process_group()


__all__ = ["app"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
