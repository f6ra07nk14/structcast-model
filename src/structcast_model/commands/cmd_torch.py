"""PyTorch related commands for the StructCast Model CLI application."""

from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from structcast.utils.base import load_yaml_from_string
from structcast.utils.security import configure_security
from typer import Argument, Option, Typer

from structcast_model.base_trainer import BaseInfo, BestCriterion, Callback, get_dataset_size
from structcast_model.commands.utils import bool_or_dict_parser, dict_parser, reduce_dict, tensor_shape_parser

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
        model, inputs, _ = torch_trainer.initial_model(_instantiate(model_pattern), shapes)
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
        model, inputs, _ = torch_trainer.initial_model(_instantiate(model_pattern), shapes)
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
        parser=bool_or_dict_parser,
        help="Whether to use EMA (Exponential Moving Average) for the model during training. "
        "Can be set to true/false or a dictionary of keyword arguments for the EMA wrapper (e.g., decay rate).",
    ),
    ema_device: str | None = Option(
        None, help="Device for the EMA model. If not specified, it will use the same device as the main model."
    ),
    loss_pattern: dict[str, Any] = Option(
        ...,
        "--loss",
        "-l",
        parser=dict_parser,
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
    metric_pattern: dict[str, Any] | None = Option(
        None,
        "--metric",
        "-m",
        parser=dict_parser,
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
    backward_pattern: dict[str, Any] | None = Option(
        ...,
        "--backward",
        "-b",
        parser=dict_parser,
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
        parser=bool_or_dict_parser,
        help='Whether to compile the model using "torch.compile". '
        'Can be set to true/false or a dictionary of keyword arguments for "torch.compile".',
    ),
    epochs: int = Option(1, "--epochs", "-e", help="Number of training epochs."),
    start_epoch: int = Option(1, help="Starting epoch number."),
    training_dataset_pattern: dict[str, Any] = Option(
        ...,
        "--training-dataset",
        "-t",
        parser=dict_parser,
        help="The object pattern used to instantiate the training dataset. "
        "For example, if the dataset is defined as `my_package.MyDataset(...)`, then the pattern should be "
        '"[_obj_, {_addr_: my_package.MyDataset, _file_: my_package.py}, {_call_: {...}}]" or '
        '"[_obj_, [_addr_, my_package.MyDataset, my_package.py], {_call_: {...}}]".',
    ),
    validation_dataset_pattern: dict[str, Any] | None = Option(
        None,
        "--validation-dataset",
        "-v",
        parser=dict_parser,
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
    seed: int = Option(0, help="Random seed for reproducibility."),
    matmul_precision: Literal["highest", "high", "medium"] = Option("high", help="Matrix multiplication precision."),
    experiment: str = Option("experiment", "--experiment", "-E", help="Experiment name for MLflow logging."),
    log_arguments: list[dict] | None = Option(
        None, "--log-arguments", "-K", parser=dict_parser, help="Additional arguments to log in MLflow."
    ),
    log_artifacts: list[Path] | None = Option(None, "--log-artifacts", "-A", help="Artifacts to log in MLflow."),
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
    device = torch_trainer.get_torch_device(device)

    def _compile_fn(module: torch.nn.Module, compile_kw: dict[str, Any] | None) -> torch.nn.Module:
        return module if compile_kw is None else torch.compile(module, **compile_kw)

    def _init_models(patterns: list[dict[str, Any]]) -> OrderedDict[str, torch.nn.Module]:
        res = OrderedDict[str, torch.nn.Module]()
        for raw in patterns:
            if len(raw) != 1:
                raise ValueError(f"Each model pattern should contain exactly one model definition. Got: {raw}")
            model_name, ptn = list(raw.items())[0]
            res[model_name] = _instantiate(ptn)
        return res

    def _check_module_outputs(module: torch.nn.Module, default: list[str] | None, name: str) -> list[str]:
        if hasattr(module, "outputs"):
            return module.outputs
        if default:
            return default
        raise ValueError(
            f'Module "{name}" does not have an "outputs" attribute. '
            f'Please provide default outputs using the "--{name}-outputs" option.'
        )

    def _to_numpy(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _to_numpy(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_numpy(v) for v in value]
        if torch.is_tensor(value):
            value = value.detach().numpy() if device == "cpu" else value.detach().cpu().numpy()
            return value if value.shape else value.reshape(1)
        return value

    def _log_criteria(info: BaseInfo, **_: torch.nn.Module) -> str:
        """Format the criteria."""
        lrs = getattr(cast(torch_trainer.TorchTrainer, info).backward, "learning_rates", None) or {}
        logs = info.logs()
        values = "\n\t".join([f"{k}: {v:.4e}" for k, v in {**logs, **lrs}.items()])
        mlflow.log_metrics({**logs, **lrs}, step=info.epoch)
        return f"Epoch {info.epoch}:\n\t{values}"

    compile_fn = partial(_compile_fn, compile_kw=instantiator.instantiate(compile_pattern))
    training_dataset = _instantiate(training_dataset_pattern)
    validation_dataset = _instantiate(validation_dataset_pattern) if validation_dataset_pattern else None
    print("Count the dataset sizes...")
    steps_per_epoch = get_dataset_size(training_dataset)
    validation_steps = 0 if validation_dataset is None else get_dataset_size(validation_dataset)
    print(f"Training dataset size: {steps_per_epoch} steps.")
    print(f"Validation dataset size: {validation_steps} steps.")
    with torch.device(device):
        models, inputs, outputs = torch_trainer.initial_model(_init_models(model_patterns), input_shapes)
        loss = compile_fn(_instantiate(loss_pattern))
        metric = compile_fn(_instantiate(metric_pattern)) if metric_pattern else None
        backward = _instantiate(backward_pattern)(**models)
        loss_outputs = _check_module_outputs(loss, loss_outputs, "loss")
        metric_outputs = _check_module_outputs(metric, metric_outputs, "metric") if metric else None
        tracker = torch_trainer.TorchTracker.from_criteria(loss_outputs, metric_outputs, compile_fn)
    mixed_precision_type = getattr(backward, "mixed_precision_type", mixed_precision_type)
    signatures = {
        n: mlflow.models.infer_signature(
            _to_numpy({k: inputs[k] for k in getattr(models[n], "inputs", list(inputs))}),
            _to_numpy(o),
        )
        for n, o in outputs.items()
    }

    def _save_fn(info: BaseInfo, suffix: str, **kwargs: torch.nn.Module) -> None:
        for name, signature in signatures.items():
            mlflow.pytorch.log_model(kwargs[name], name=f"{name}{suffix}", step=info.epoch, signature=signature)

    def _create_criterion_callback(name: str, mode: Literal["min", "max"]) -> Callback[torch.nn.Module]:
        on_best: list[Any] = [lambda i, t, v, **m: mlflow.log_metric(f"{t}_best", v, step=i.epoch)]  # type: ignore[arg-type]
        if name in save_criteria:
            on_best.append(lambda i, t, v, **m: _save_fn(i, suffix="_best", **m))  # type: ignore[arg-type]
        return BestCriterion[torch.nn.Module](target=name, mode=mode, on_best=on_best)

    autocast = torch_trainer.get_autocast(mixed_precision_type, device)
    step_kw = {"models": list(models), "losses": loss, "metrics": metric, "autocast": autocast}
    trainer = torch_trainer.TorchTrainer(
        inference_wrapper=None
        if ema is None
        else torch_trainer.TimmEmaWrapper.from_models(
            models,
            callbacks=[partial(_save_fn, suffix="_ema")],
            device=torch.device(ema_device or device),
            **instantiator.instantiate(ema),
        ),
        training_step=torch_trainer.TrainingStep(**step_kw),
        validation_step=torch_trainer.ValidationStep(**step_kw),
        backward=backward,
        tracker=tracker,
    )
    if ci:
        trainer.on_epoch_end.append(lambda i, **_: print(_log_criteria(i)))  # type: ignore[arg-type]
    else:
        pbar = tqdm.tqdm(unit="batch")

        def _update_criteria(info: BaseInfo, criteria: list[str], **_: torch.nn.Module) -> None:
            logs = info.logs()
            pbar.update()
            pbar.set_postfix([(n, logs[n]) for n in criteria])

        trainer.on_training_begin.append(lambda **_: pbar.reset(steps_per_epoch))  # type: ignore[arg-type]
        train_losses = [f"{trainer.training_prefix}{n}" for n in loss_outputs]
        trainer.on_training_step_end.append(partial(_update_criteria, criteria=train_losses))  # type: ignore[arg-type]
        trainer.on_training_end.append(lambda **_: pbar.refresh())  # type: ignore[arg-type]
        trainer.on_validation_begin.append(lambda **_: pbar.reset(validation_steps))  # type: ignore[arg-type]
        valid_losses = [f"{trainer.validation_prefix}{n}" for n in loss_outputs]
        trainer.on_validation_step_end.append(partial(_update_criteria, criteria=valid_losses))  # type: ignore[arg-type]
        trainer.on_validation_end.append(lambda **_: pbar.refresh())  # type: ignore[arg-type]
        trainer.on_epoch_end.append(lambda i, **_: pbar.write(_log_criteria(i)))  # type: ignore[arg-type]
    trainer.on_epoch_end.append(partial(_save_fn, suffix=""))
    trainer.on_epoch_end += [_create_criterion_callback(n, mode="max") for n in higher_criteria]
    trainer.on_epoch_end += [_create_criterion_callback(n, mode="min") for n in lower_criteria]
    model_parameters = {n: sum(p.numel() for p in m.parameters() if p.requires_grad) for n, m in models.items()}
    arguments = {
        **reduce_dict(log_arguments),
        "models": model_patterns,
        "shapes": input_shapes,
        "device": device,
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
        mlflow.log_param("parameters", model_parameters)
        if hasattr(backward, "param_group_names"):
            mlflow.log_dict(backward.param_group_names, "param_groups.yaml")
        mlflow.log_dict(arguments, "arguments.yaml")
        for artifact in log_artifacts or []:
            mlflow.log_artifact(str(artifact))
        trainer.fit(
            epochs=epochs,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            start_epoch=start_epoch,
            validation_frequency=validation_frequency,
            **models,
        )


__all__ = ["app"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
