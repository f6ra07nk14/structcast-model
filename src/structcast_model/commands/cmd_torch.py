"""PyTorch related commands for the StructCast Model CLI application."""

from collections import OrderedDict
from functools import partial
import inspect
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from time import time
from typing import TYPE_CHECKING, Any, Literal

from structcast.utils.base import dump_yaml_to_string
from typer import Argument, Option, Typer

from structcast_model.base_trainer import (
    Printer,
    ProgressBar,
    SimpleDataProvider,
)
from structcast_model.commands.utils import (
    bool_or_path_or_dict_parser,
    dict_parser,
    instantiate_object,
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
    import wandb

    from structcast_model.builders import torch_builder
    from structcast_model.torch import (
        distributed as torch_distributed,
        logger as torch_logger,
        mlflow_logger,
        trainer as torch_trainer,
        wandb_logger,
    )
    import torch
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    calflops = LazyModuleImporter("calflops")
    mlflow = LazyModuleImporter("mlflow")
    np = LazyModuleImporter("numpy")
    ptflops = LazyModuleImporter("ptflops")
    wandb = LazyModuleImporter("wandb")
    instantiator = LazyModuleImporter("structcast.core.instantiator")
    torch_builder = LazyModuleImporter("structcast_model.builders.torch_builder")
    torch_distributed = LazyModuleImporter("structcast_model.torch.distributed")
    torch_logger = LazyModuleImporter("structcast_model.torch.logger")
    mlflow_logger = LazyModuleImporter("structcast_model.torch.mlflow_logger")
    torch_trainer = LazyModuleImporter("structcast_model.torch.trainer")
    wandb_logger = LazyModuleImporter("structcast_model.torch.wandb_logger")
    torch = LazyModuleImporter("torch")


app = Typer(no_args_is_help=True)
creator = Typer(no_args_is_help=True)
app.add_typer(creator, name="create", help="Commands for creating PyTorch models and learner classes.")

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
compile_pattern: dict[str, Any] | None = Option(
    None,
    "--compile",
    "-c",
    parser=bool_or_path_or_dict_parser,
    help='Whether to compile the model using "torch.compile". '
    'Can be set to true/false, a path to a YAML file, or a dictionary of keyword arguments for "torch.compile".',
)
matmul_precision: Literal["highest", "high", "medium"] = Option(
    "high", envvar="MATMUL_PRECISION", help="Matrix multiplication precision."
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


@creator.command(name="learner")
def create_learner(
    cfg_path: str = Argument(..., help="Path to the learner configuration file."),
    output: str | None = output_script_path,
    parameters: list[dict] | None = template_param,
    classname: str = Option("Learner", "--classname", "-c", help="Name the learner class."),
) -> None:
    """Create a PyTorch learner class from the given configuration file and parameters."""
    builder = torch_builder.TorchLearnerBuilder.from_path(cfg_path)
    builder(parameters=reduce_dict(parameters), classname=classname)(output)


def _compile_module(module: Any, compile_kw: dict[str, Any] | None) -> Any:
    """Compile a PyTorch module if compile_kw is provided."""
    return module if compile_kw is None else torch.compile(module, **compile_kw)


def _instantiate_models(patterns: list[dict]) -> "OrderedDict[str, Any]":
    """Instantiate models from a list of name-pattern mappings."""
    res: OrderedDict[str, Any] = OrderedDict()
    for raw in patterns:
        if len(raw) != 1:
            raise ValueError(f"Each model pattern should contain exactly one model definition. Got: {raw}")
        model_name, ptn = next(iter(raw.items()))
        res[model_name] = instantiate_object(ptn)
    return res


def _get_module_outputs(module: Any, default: list[str] | None, name: str) -> list[str]:
    """Return output names from a module attribute or the provided default, raising if neither is available."""
    if default:
        return default
    if hasattr(module, "outputs"):
        return module.outputs
    raise ValueError(
        f'Module "{name}" does not have an "outputs" attribute. '
        f'Please provide default outputs using the "--{name}-outputs" option.'
    )


def _fetch_training_state(reference: str) -> dict[str, Any]:
    """Load a saved training state from a local path, an MLflow `runs:/` URI, or a `wandb://` reference.

    Args:
        reference (str): The training state location: a local path, `runs:/<run_id>/<artifact>`, or
            `wandb://<entity>/<project>/<run_id>/<file>`.

    Returns:
        dict[str, Any]: The loaded training state.

    Raises:
        ValueError: If a downloaded MLflow artifact directory holds no state file.
    """
    if reference.startswith("runs:/"):
        path = Path(mlflow.artifacts.download_artifacts(artifact_uri=reference))
        if path.is_dir():
            # `mlflow.pytorch.log_state_dict` writes the tensors to a file inside the artifact directory.
            states = sorted(path.glob("*.pth"))
            if not states:
                raise ValueError(f'No "*.pth" training state found in the downloaded MLflow artifact "{path}".')
            path = states[0]
    elif reference.startswith("wandb://"):
        entity, project, run_id, filename = reference.removeprefix("wandb://").split("/", 3)
        with TemporaryDirectory() as directory:
            wandb.Api().run(f"{entity}/{project}/{run_id}").file(filename).download(root=directory, replace=True)
            # The download is deleted with the temporary directory, so it is read inside the block.
            return torch.load(Path(directory) / filename, map_location="cpu", weights_only=True)
    else:
        path = Path(reference)
    # `weights_only` because the reference is user input, and an unpickled checkpoint executes code.
    return torch.load(path, map_location="cpu", weights_only=True)


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
    matmul_precision: Literal["highest", "high", "medium"] = matmul_precision,
) -> None:
    """Measure the average inference time of a PyTorch model."""
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision(matmul_precision)
    device = torch_trainer.get_torch_device(device)
    print("Initializing the model...")
    with torch.device(device):
        model = instantiate_object(model_pattern)
        shapes = torch_trainer.resolve_input_shapes(model, shapes)
        torch_trainer.initial_model(model, shapes)
    print("Skipping compilation..." if compile_pattern is None else "Compiling the model...")
    model = _compile_module(model, instantiator.instantiate(compile_pattern))
    if training_mode:
        model.train()
    else:
        model.eval()
    cuda_sync = torch.cuda.synchronize if "cuda" in device else lambda: None
    device_type = torch_trainer.get_torch_device_type(device)

    def _measure_single_run() -> float:
        with torch.device(device):
            inputs = torch_trainer.create_torch_inputs(shapes, batch_size=batch_size)
        start_time = time()
        with torch_trainer.autocast_inputs(inputs, device_type):
            model(**inputs)
        cuda_sync()
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
    device = torch_trainer.get_torch_device(device)
    with torch.device(device):
        model = instantiate_object(model_pattern)
        inputs, _ = torch_trainer.initial_model(model, shapes)
        with torch_trainer.autocast_inputs(inputs, torch_trainer.get_torch_device_type(device)):
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
    device = torch_trainer.get_torch_device(device)
    with torch.device(device):
        model = instantiate_object(model_pattern)
        inputs, _ = torch_trainer.initial_model(model, shapes)
        with torch_trainer.autocast_inputs(inputs, torch_trainer.get_torch_device_type(device)):
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
def train(  # noqa: PLR0912,PLR0913,PLR0915
    model_patterns: list[dict] = Argument(
        parser=dict_parser,
        help="The object patterns used to instantiate models. "
        "For example, if the model is defined as `model_name = my_package.MyModel(...)`, then the pattern should be "
        '"model_name: [_obj_, {_addr_: my_package.MyModel, _file_: my_package.py}, {_call_: {...}}]" or '
        '"model_name: [_obj_, [_addr_, my_package.MyModel, my_package.py], {_call_: {...}}]".',
    ),
    initializer_patterns: list[dict] | None = Option(
        None,
        "--initializer",
        "-I",
        parser=dict_parser,
        help="The object patterns used to instantiate initializers for the models. "
        "For example, if the initializer is defined as `my_package.initialize_fn`, then the pattern should be "
        '"model_name: [_obj_, {_addr_: my_package.initialize_fn, _file_: my_package.py}]" or '
        '"model_name: [_obj_, [_addr_, my_package.initialize_fn, my_package.py]]".',
    ),
    shapes: list[dict] | None = shapes,
    device: str | None = device,
    learner_pattern: Any = Option(
        ...,
        "--learner",
        "-L",
        parser=path_or_any_parser,
        help="The object pattern used to instantiate the learner class. "
        "For example, if the learner class is defined as `my_package.MyLearner(...)`, then the pattern should be "
        '"[_obj_, {_addr_: my_package.MyLearner, _file_: my_package.py}, {_call_: {...}}]" or '
        '"[_obj_, [_addr_, my_package.MyLearner, my_package.py], {_call_: {...}}]".',
    ),
    learner_outputs: list[str] | None = Option(
        None,
        "--learner-outputs",
        "-LO",
        help="Default outputs for the learner module if it doesn't have an 'outputs' attribute.",
    ),
    compile_pattern: dict[str, Any] | None = compile_pattern,
    trainer_pattern: Any | None = Option(
        None,
        "--trainer",
        parser=path_or_any_parser,
        help="The object pattern used to instantiate the trainer. "
        "For example, if the trainer is defined as `my_package.MyTrainer`, then the pattern should be "
        '"[_obj_, {_addr_: my_package.MyTrainer, _file_: my_package.py}]" or '
        '"[_obj_, [_addr_, my_package.MyTrainer, my_package.py]]".',
    ),
    epochs: int = Option(1, "--epochs", "-e", help="Number of training epochs."),
    start_epoch: int = Option(1, help="Starting epoch number."),
    resume: str | None = Option(
        None,
        "--resume",
        help="Training state to resume from: a local path, an MLflow 'runs:/<run_id>/<artifact>' URI, "
        "or 'wandb://<entity>/<project>/<run_id>/<file>'. "
        "Restores models, optimizers, grad scalers, and continues from the saved epoch.",
    ),
    training_dataset_pattern: Any = Option(
        ...,
        "--training-dataset",
        parser=path_or_any_parser,
        help="The object pattern used to instantiate the training dataset. "
        "For example, if the dataset is defined as `my_package.MyDataset(...)`, then the pattern should be "
        '"[_obj_, {_addr_: my_package.MyDataset, _file_: my_package.py}, {_call_: {...}}]" or '
        '"[_obj_, [_addr_, my_package.MyDataset, my_package.py], {_call_: {...}}]".',
    ),
    validation_dataset_pattern: Any | None = Option(
        None,
        "--validation-dataset",
        "-V",
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
    matmul_precision: Literal["highest", "high", "medium"] = matmul_precision,
    experiment: str = Option(
        "experiment", "--experiment", "-E", envvar="EXPERIMENT", help="Experiment name for the logger."
    ),
    logger_name: Literal["mlflow", "wandb"] = Option(
        "mlflow", "--logger", help="Experiment tracking service to record the run to."
    ),
    log_arguments: list[dict] | None = Option(
        None, "--log-arguments", "-K", parser=dict_parser, help="Additional arguments to log."
    ),
    log_artifacts: list[Path] | None = Option(None, "--log-artifacts", "-A", help="Artifacts to log."),
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
    strategy_pattern: Any | None = Option(
        None,
        "--strategy",
        parser=path_or_any_parser,
        help="Object pattern instantiating a distributed strategy factory; called with device=... and local_rank=.... "
        "Defaults to DistributedDataParallelStrategy when a distributed environment is detected, "
        "else SingleDeviceStrategy.",
    ),
) -> None:
    """Train a PyTorch model, recording the run to an experiment tracking service."""
    if not model_patterns:
        raise ValueError("At least one model pattern must be provided.")
    device, global_rank, local_rank, world_size, distributed = torch_distributed.initial_distributed_env(
        device=device, dist_backend=dist_backend, dist_url=dist_url, return_dict=False
    )
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision(matmul_precision)
    torch.manual_seed(seed + global_rank)
    np.random.seed(seed + global_rank)
    random.seed(seed + global_rank)
    if strategy_pattern is not None:
        strategy = instantiate_object(strategy_pattern)(device=device, local_rank=local_rank)
    elif distributed:
        strategy = torch_distributed.DistributedDataParallelStrategy(device=device, local_rank=local_rank)
    else:
        strategy = torch_distributed.SingleDeviceStrategy(device=device, local_rank=local_rank)
    is_main = global_rank == 0
    input_shapes = reduce_dict(shapes)
    initializers = instantiator.instantiate(reduce_dict(initializer_patterns))
    compile_fn = partial(_compile_module, compile_kw=instantiator.instantiate(compile_pattern))
    training_dataset = instantiate_object(training_dataset_pattern)
    validation_dataset = instantiate_object(validation_dataset_pattern) if validation_dataset_pattern else None
    provider = SimpleDataProvider(training_dataset=training_dataset, validation_dataset=validation_dataset)
    if is_main:
        print("Count the dataset sizes...")
    if is_main:
        print(f"Training dataset size: {provider.steps_per_epoch} steps.")
        print(f"Validation dataset size: {provider.validation_steps} steps.")
    # Everything below runs on the training device: the models, and the tracker buffers, which are
    # allocated with torch.zeros and would otherwise fail the first step mixing CUDA criteria with
    # CPU buffers.
    with torch.device(device):
        models = _instantiate_models(model_patterns)
        input_shapes = torch_trainer.resolve_input_shapes(models, input_shapes) or {}
        torch_trainer.initial_model(models, input_shapes)
        # A resumed run loads its weights below, which would overwrite whatever the initializers and
        # the initial-weight broadcast produce here.
        if is_main and resume is None:
            for model_name, model in models.items():
                if model_name in initializers:
                    model.apply(initializers[model_name])
        if resume is None:
            strategy.sync_initial_weights(models)
        models = OrderedDict((n, compile_fn(m)) for n, m in models.items())
        models = strategy.wrap(models)
        factory = instantiate_object(learner_pattern)
        # Only learners declaring the parameter get the strategy's scaler creator: a learner taking
        # its models as **kwargs would otherwise record the creator as one more model.
        try:
            takes_scaler_creator = "__grad_scaler_creator__" in inspect.signature(factory).parameters
        except (TypeError, ValueError):  # Callables implemented in C expose no signature.
            takes_scaler_creator = False
        if takes_scaler_creator:
            learner = factory(**models, __grad_scaler_creator__=strategy.grad_scaler_creator)
        else:
            learner = factory(**models)
        learner_outputs = _get_module_outputs(learner, learner_outputs, "learner")
        tracker = torch_trainer.TorchTracker.from_criteria(learner_outputs, compile_fn, distributed)
    # The flow functions are the compile units; the step itself stays eager. See ADR-0004.
    if hasattr(learner, "flow_functions"):
        for flow_name in list(learner.flow_functions):
            setattr(learner, flow_name, compile_fn(getattr(learner, flow_name)))
    if resume is not None:
        raw_state = _fetch_training_state(resume) if is_main else None
        state = strategy.load_state_dict(
            models, getattr(learner, "optimizers", {}), getattr(learner, "optimizer_models", None), raw_state
        )
        for scaler_name, scaler in getattr(learner, "grad_scalers", {}).items():
            if state.get("grad_scalers", {}).get(scaler_name):
                scaler.load_state_dict(state["grad_scalers"][scaler_name])
        resumed_epoch = state["meta"]["epoch"] + 1
        if start_epoch != 1 and is_main:
            print(f"Ignoring --start-epoch {start_epoch}: the resumed state continues at epoch {resumed_epoch}.")
        start_epoch = resumed_epoch
    trainer_type = torch_trainer.TorchTrainer if trainer_pattern is None else instantiate_object(trainer_pattern)
    trainer = trainer_type(device=device, learner=learner, tracker=tracker, data=provider, callbacks=[])
    if is_main:
        logger_type = mlflow_logger.MLflowLogger if logger_name == "mlflow" else wandb_logger.WandbLogger
        logger: torch_logger.Logger = logger_type(experiment=experiment)
    else:
        logger = torch_logger.NullLogger()
    # The saver and the best-criterion monitors run collectives, so they are built on every rank;
    # only rank 0 holds a real logger and writes anything. See ADR-0005.
    saver = torch_trainer.TrainingStateSaver(logger=logger, strategy=strategy)
    bests = torch_trainer.TorchBestCriterion.from_criteria(
        higher_criteria, lower_criteria, save_criteria, logger=logger, strategy=strategy
    )
    display: list[Any] = []
    if is_main:
        display.append(
            Printer()
            if ci
            else ProgressBar(
                steps_per_epoch=provider.steps_per_epoch,
                validation_steps=provider.validation_steps,
                training_criteria=[f"{trainer.training_prefix}{n}" for n in learner_outputs],
                validation_criteria=[f"{trainer.validation_prefix}{n}" for n in learner_outputs],
            )
        )
    trainer.callbacks = [*display, logger, saver, *bests]
    arguments = {
        **reduce_dict(log_arguments),
        "models": model_patterns,
        "parameters": {n: sum(p.numel() for p in m.parameters() if p.requires_grad) for n, m in models.items()},
        "initializers": initializer_patterns,
        "shapes": input_shapes,
        "device": device,
        "distributed": distributed,
        "world_size": world_size,
        "learner": learner_pattern,
        "learner_outputs": learner_outputs,
        "compile": compile_pattern,
        "trainer": trainer_pattern,
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
        "logger": logger_name,
        "ci": ci,
    }
    try:
        # One path for every rank: the NullLogger ranks run the same lifecycle and discard it all.
        with logger:
            logger.log_params(
                {
                    "cuda_version": torch.version.cuda,
                    "torch_version": torch.__version__,
                    "epochs": epochs,
                    "steps_per_epoch": provider.steps_per_epoch,
                    "validation_steps": provider.validation_steps,
                }
            )
            logger.log_dict(arguments, "arguments.yaml")
            if hasattr(learner, "param_group_names"):
                logger.log_dict(learner.param_group_names, "param_groups.yaml")
            for artifact in log_artifacts or []:
                logger.log_artifact(str(artifact))
            if is_main:
                print(f"Registered callbacks:\n{dump_yaml_to_string(trainer.describe())}")
            trainer.fit(epochs=epochs, start_epoch=start_epoch, validation_frequency=validation_frequency)
    finally:
        if distributed:
            torch.distributed.destroy_process_group()


__all__ = ["app"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
