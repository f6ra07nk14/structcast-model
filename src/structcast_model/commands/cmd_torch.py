"""PyTorch related commands for the StructCast Model CLI application."""

from collections import OrderedDict
from functools import partial
import inspect
from pathlib import Path
import random
from time import time
from typing import TYPE_CHECKING, Any, Literal

from structcast.utils.base import dump_yaml_to_string
from typer import Argument, Option, Typer

# Both are package shims routing to lazy submodules, so importing them pulls in no framework.
# Wrapping them in `LazyModuleImporter` would not work: it copies the shim's still unresolved
# submodule slots, so every access after the first would hand back `None`.
from structcast_model import loggers as scm_loggers, torch as scm_torch
from structcast_model.base_trainer import (
    Printer,
    ProgressBar,
    SimpleDataProvider,
)
from structcast_model.commands.shared_args import (
    PATH_FORM_HELP,
    batch_size,
    compile_option,
    model_pattern,
    object_pattern_help,
    output_script_path,
    shapes_help,
    shapes_option,
    template_param_option,
    times,
    training_mode,
    warmup_runs,
)
from structcast_model.commands.utils import (
    dict_parser,
    instantiate_object,
    path_or_any_parser,
    reduce_dict,
)

if TYPE_CHECKING:
    import calflops
    import numpy as np
    import ptflops
    from structcast.core import instantiator

    from structcast_model.builders import torch as torch_builder
    import torch
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    calflops = LazyModuleImporter("calflops")
    np = LazyModuleImporter("numpy")
    ptflops = LazyModuleImporter("ptflops")
    instantiator = LazyModuleImporter("structcast.core.instantiator")
    torch_builder = LazyModuleImporter("structcast_model.builders.torch")
    torch = LazyModuleImporter("torch")


app = Typer(no_args_is_help=True)
creator = Typer(no_args_is_help=True)
app.add_typer(creator, name="create", help="Commands for creating PyTorch models and learner classes.")

DEVICE_HELP = (
    'Computation device to use: "cpu", "cuda", or an indexed form such as "cuda:1". '
    'If not specified, "cuda" is used when available, otherwise "cpu"; an explicitly requested CUDA device '
    "falls back to CPU with a warning when CUDA is unavailable."
)

SHAPES_HELP = shapes_help('"image: [3, 224, 224]"', "torch.zeros")
template_param = template_param_option(
    'For example: --parameter "model: {input_size: 128, output_size: 10}" --parameter "optimizer: {lr: 0.001}"'
)
# --shape and --device read differently under `train`, so the commands share only the prose that is true for both.
shapes = shapes_option(
    SHAPES_HELP + " When omitted, the INPUT_SHAPES declared by the built model are used, and the run fails only when "
    "neither exists."
)
device = Option(None, "--device", "-d", help=DEVICE_HELP)
compile_pattern: dict[str, Any] | None = compile_option("torch.compile")
matmul_precision: Literal["highest", "high", "medium"] = Option(
    "high",
    envvar="MATMUL_PRECISION",
    help='Precision for float32 matrix multiplications: "highest" keeps full float32, while "high" and "medium" '
    "trade accuracy for tensor-core speed.",
)


@creator.command(name="model")
def create_model(
    cfg_path: str = Argument(..., help="Path to the model configuration file."),
    output: str | None = output_script_path,
    parameters: list[dict] | None = template_param,
    classname: str = Option("Model", "--classname", "-n", help="Name of the generated model class."),
    structured_output: bool | None = Option(
        None,
        "--structured-output/--no-structured-output",
        help="Force dict (structured) output on the root model. By default the configuration's "
        "STRUCTURED_OUTPUT decides, which is false unless set. Ignored with --sublayer: the "
        "selected layer's own configuration decides.",
    ),
    sublayer: str | None = Option(
        None, "--sublayer", help="The reference to a sublayer in the template to build instead of the root layer."
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
    classname: str = Option("Learner", "--classname", "-n", help="Name of the generated Learner class."),
) -> None:
    """Create a PyTorch learner class from the given configuration file and parameters."""
    builder = torch_builder.TorchLearnerBuilder.from_path(cfg_path)
    builder(parameters=reduce_dict(parameters), classname=classname)(output)


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
    matmul_precision: Literal["highest", "high", "medium"] = matmul_precision,
) -> None:
    """Measure the average inference time of a PyTorch model."""
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision(matmul_precision)
    device = scm_torch.trainer.get_torch_device(device)
    print("Initializing the model...")
    with torch.device(device):
        model = instantiate_object(model_pattern)
        shapes = scm_torch.trainer.resolve_input_shapes(model, shapes)
        scm_torch.trainer.initial_model(model, shapes)
    print("Skipping compilation..." if compile_pattern is None else "Compiling the model...")
    model = scm_torch.distributed.SingleDeviceStrategy(device=device).compile(
        model, instantiator.instantiate(compile_pattern)
    )
    if training_mode:
        model.train()
    else:
        model.eval()
    cuda_sync = torch.cuda.synchronize if "cuda" in device else lambda: None
    device_type = scm_torch.trainer.get_torch_device_type(device)

    def _measure_single_run() -> float:
        with torch.device(device):
            inputs = scm_torch.trainer.create_torch_inputs(shapes, batch_size=batch_size)
        start_time = time()
        with scm_torch.trainer.autocast_inputs(inputs, device_type):
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
    flops_units: Literal["GMac", "MMac", "KMac"] = Option(
        "GMac", help="Unit for the reported multiply-accumulate count."
    ),
    param_units: Literal["M", "K", "B"] = Option(
        "M", help="Units for parameters: M (millions), K (thousands), or B (billions)."
    ),
    backend: Literal["pytorch", "aten"] = Option(
        "aten",
        help='Backend for the FLOPs computation. Do not use the "pytorch" backend for transformer architectures.',
    ),
    device: str | None = device,
) -> None:
    """Calculate the FLOPs and number of parameters of a PyTorch model using ptflops."""
    device = scm_torch.trainer.get_torch_device(device)
    with torch.device(device):
        model = instantiate_object(model_pattern)
        inputs, _ = scm_torch.trainer.initial_model(model, shapes)
        with scm_torch.trainer.autocast_inputs(inputs, scm_torch.trainer.get_torch_device_type(device)):
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
    device = scm_torch.trainer.get_torch_device(device)
    with torch.device(device):
        model = instantiate_object(model_pattern)
        inputs, _ = scm_torch.trainer.initial_model(model, shapes)
        with scm_torch.trainer.autocast_inputs(inputs, scm_torch.trainer.get_torch_device_type(device)):
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


def _resolve_strategy(
    strategy_pattern: Any, device: str, local_rank: int, distributed: bool
) -> "scm_torch.distributed.DistributedStrategy":
    """Resolve the run's strategy: an explicit pattern wins, then DDP when distributed, else single-device."""
    if strategy_pattern is not None:
        return instantiate_object(strategy_pattern)(device=device, local_rank=local_rank)
    if distributed:
        return scm_torch.distributed.DistributedDataParallelStrategy(device=device, local_rank=local_rank)
    return scm_torch.distributed.SingleDeviceStrategy(device=device, local_rank=local_rank)


def _assemble_learner(
    *,
    model_patterns: list[dict],
    input_shapes: dict[str, Any],
    initializers: dict[str, Any],
    resume: str | None,
    strategy: "scm_torch.distributed.DistributedStrategy",
    compile_kw: dict[str, Any] | None,
    learner_pattern: Any,
    learner_outputs: list[str] | None,
    device: str,
    distributed: bool,
    is_main: bool,
) -> tuple["OrderedDict[str, torch.nn.Module]", Any, list[str], Any]:
    """Instantiate, initialize, compile and wrap the models, then build the learner and its tracker."""
    # Everything below runs on the training device: the models, and the tracker buffers, which are
    # allocated with torch.zeros and would otherwise fail the first step mixing CUDA criteria with
    # CPU buffers.
    with torch.device(device):
        models = _instantiate_models(model_patterns)
        input_shapes = scm_torch.trainer.resolve_input_shapes(models, input_shapes) or {}
        scm_torch.trainer.initial_model(models, input_shapes)
        # A resumed run loads its weights later, which would overwrite whatever the initializers and
        # the initial-weight broadcast produce here.
        if is_main and resume is None:
            for model_name, model in models.items():
                if model_name in initializers:
                    model.apply(initializers[model_name])
        if resume is None:
            strategy.sync_initial_weights(models)
        models = OrderedDict((n, strategy.compile(m, compile_kw)) for n, m in models.items())
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
        tracker = scm_torch.trainer.TorchTracker.from_criteria(
            learner_outputs, partial(strategy.compile, compile_kw=compile_kw), distributed
        )
    # The flow functions are the compile units; the step itself stays eager. See ADR-0004.
    # Flow functions compile only on a single device: distributed wrappers graph-break inside the
    # flow, and the fragment overhead measurably exceeds the glue-fusion gain (H200 numbers in
    # docs/references/flow-compile-step-time-h200.md). The models themselves compile either way.
    if hasattr(learner, "flow_functions") and not distributed:
        for flow_name in list(learner.flow_functions):
            setattr(learner, flow_name, strategy.compile(getattr(learner, flow_name), compile_kw))
    return models, learner, learner_outputs, tracker


def _restore_training_state(
    *,
    resume: str,
    strategy: "scm_torch.distributed.DistributedStrategy",
    models: "OrderedDict[str, torch.nn.Module]",
    learner: Any,
    start_epoch: int,
    is_main: bool,
    logger: "scm_loggers.base.Logger",
) -> int:
    """Load the resumed state into models, optimizers and scalers; the saved epoch wins over --start-epoch.

    The logger owns the reference format, and only rank 0 holds a real one: the `NullLogger` ranks
    fetch nothing and take the state from the strategy's broadcast.
    """
    raw_state = logger.fetch_training_state(resume)
    state = strategy.load_state_dict(models, learner.optimizers, learner.optimizer_models, raw_state)
    for scaler_name, scaler in getattr(learner, "grad_scalers", {}).items():
        if state.get("grad_scalers", {}).get(scaler_name):
            scaler.load_state_dict(state["grad_scalers"][scaler_name])
    resumed_epoch = state["meta"]["epoch"] + 1
    if start_epoch != 1 and is_main:
        print(f"Ignoring --start-epoch {start_epoch}: the resumed state continues at epoch {resumed_epoch}.")
    return resumed_epoch


def _build_callbacks(
    *,
    trainer: Any,
    provider: SimpleDataProvider,
    strategy: "scm_torch.distributed.DistributedStrategy",
    learner_outputs: list[str],
    higher_criteria: list[str],
    lower_criteria: list[str],
    save_criteria: list[str],
    logger: "scm_loggers.base.Logger",
    ci: bool,
    is_main: bool,
) -> None:
    """Install the logger and the saver/best/display callbacks on the trainer."""
    # The saver and the best-criterion monitors run collectives, so they are built on every rank;
    # only rank 0 holds a real logger and writes anything. See ADR-0005.
    saver = scm_torch.trainer.TrainingStateSaver(logger=logger, strategy=strategy)
    bests = scm_torch.trainer.TorchBestCriterion.from_criteria(
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


@app.command()
def train(  # noqa: PLR0913, PLR0917  # The CLI surface: every training option is one Typer parameter.
    model_patterns: list[dict] = Argument(
        parser=dict_parser,
        help=object_pattern_help("the model", "MyModel", keyed=True)
        + ' Pass one positional argument per model, each a mapping with exactly one "name: pattern" entry given '
        "inline; a file path is not accepted here. The names are passed to the --learner factory as keyword "
        "arguments and are the keys used by --initializer.",
    ),
    initializer_patterns: list[dict] | None = Option(
        None,
        "--initializer",
        "-I",
        parser=dict_parser,
        help=object_pattern_help("the initializer", "initialize_fn", keyed=True, call=False)
        + " Key each pattern by one of the model names given as positional arguments; a key matching no model is "
        "ignored. Each initializer is applied to every submodule of its model on rank 0 and broadcast to the other "
        "ranks, and the whole option is skipped when --resume is given, because the loaded state would overwrite it.",
    ),
    shapes: list[dict] | None = shapes_option(
        SHAPES_HELP
        + " Repeat the option to declare more inputs; occurrences are merged at the top level, so an input named "
        "twice keeps only the last occurrence. When omitted, the INPUT_SHAPES declared by the built models are "
        "used, merged across them."
    ),
    device: str | None = Option(
        None,
        "--device",
        "-d",
        help=DEVICE_HELP + " Under a distributed launch the CUDA index is replaced by the process's LOCAL_RANK.",
    ),
    learner_pattern: Any = Option(
        ...,
        "--learner",
        "-L",
        parser=path_or_any_parser,
        help=object_pattern_help("the learner class", "MyLearner")
        + PATH_FORM_HELP
        + " The factory is called with one keyword argument per positional model pattern, so its parameter names "
        "must match those model names.",
    ),
    learner_outputs: list[str] | None = Option(
        None,
        "--learner-outputs",
        "-LO",
        help="Criterion names the Learner's steps produce; they build the tracker and the progress-bar rows. "
        "Overrides the names the Learner declares itself, and is required when it declares none. Give the "
        'unprefixed names: the criterion options then take "loss" for the training criterion and "val_loss" for '
        "the validation one.",
    ),
    compile_pattern: dict[str, Any] | None = compile_pattern,
    trainer_pattern: Any | None = Option(
        None,
        "--trainer",
        parser=path_or_any_parser,
        help=object_pattern_help("the trainer", "MyTrainer", call=False)
        + PATH_FORM_HELP
        + " The pattern must build a callable accepted as trainer(device=..., learner=..., tracker=..., data=..., "
        "callbacks=[]) whose result exposes training_prefix, validation_prefix, callbacks, describe() and "
        "fit(epochs, start_epoch, validation_frequency); subclassing TorchTrainer is the intended way.",
    ),
    epochs: int = Option(
        1,
        "--epochs",
        "-e",
        help="Epoch number to train up to, inclusive: training runs from --start-epoch (or the resumed epoch) "
        "through this number, so it is a count only when starting at epoch 1 and must not be smaller than the "
        "starting epoch. A resumed run must set it above the epoch stored in the resumed state.",
    ),
    start_epoch: int = Option(
        1,
        help="First epoch number to run, 1-based and at most --epochs. It only offsets the loop counter; no data "
        "is skipped. Ignored when --resume is given, which continues at the epoch stored in the resumed state "
        "plus one.",
    ),
    resume: str | None = Option(
        None,
        "--resume",
        help="Training state to resume from, in a form the active --logger understands: a local path always "
        "works, 'runs:/<run_id>/<artifact>' requires --logger mlflow, and "
        "'wandb://<entity>/<project>/<run_id>/<file>' requires --logger wandb; resuming across services is not "
        "supported. Restores models, optimizers, grad scalers, and continues from the saved epoch. Resume is exact "
        "only at epoch boundaries: the saved state carries the epoch, step and update counters but no data-order, "
        "sampler or RNG state, so the resumed epoch restarts from the beginning of the dataset.",
    ),
    training_dataset_pattern: Any = Option(
        ...,
        "--training-dataset",
        parser=path_or_any_parser,
        help=object_pattern_help("the training dataset", "MyDataset") + PATH_FORM_HELP,
    ),
    validation_dataset_pattern: Any | None = Option(
        None,
        "--validation-dataset",
        "-V",
        parser=path_or_any_parser,
        help=object_pattern_help("the validation dataset", "MyDataset") + PATH_FORM_HELP,
    ),
    validation_frequency: int = Option(
        1,
        "--validation-frequency",
        "-f",
        help="Run validation every N epochs; must be at least 1. On epochs where validation does not run, no "
        '"val_" criterion is produced, so val_-named best and save criteria are only monitored on validated epochs.',
    ),
    lower_criteria: list[str] = Option(
        ...,
        "--lower-criterion",
        "-LC",
        default_factory=list,
        show_default=False,
        help="Criterion names whose lower values are better, monitored for their lowest value. Name them as they "
        'appear in the epoch logs: training criteria keep the Learner\'s names, validation criteria carry the "val_" '
        'prefix (e.g. "val_loss"). A name no epoch produces is silently never monitored, and omitting the option '
        "monitors nothing.",
    ),
    higher_criteria: list[str] = Option(
        ...,
        "--higher-criterion",
        "-HC",
        default_factory=list,
        show_default=False,
        help="Criterion names whose higher values are better, monitored for their highest value. Name them as they "
        'appear in the epoch logs: training criteria keep the Learner\'s names, validation criteria carry the "val_" '
        'prefix (e.g. "val_accuracy"). A name no epoch produces is silently never monitored, and omitting the option '
        "monitors nothing.",
    ),
    save_criteria: list[str] = Option(
        ...,
        "--save-criterion",
        "-SC",
        default_factory=list,
        show_default=False,
        help='Criterion names whose best-scoring model states are saved, as a "best_<criterion>" artifact. Each '
        "name must also be given to --lower-criterion or --higher-criterion, spelled the same way; a name in "
        "neither list is silently ignored, and omitting the option saves no best-model artifact.",
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
        None,
        "--log-arguments",
        "-K",
        parser=dict_parser,
        help='Extra key-value pairs to record in the run\'s "arguments.yaml" artifact, in the format "key: value". '
        "Repeat the option to add more keys; a key given twice keeps only the last occurrence, and the keys the run "
        "records itself take precedence over keys of the same name.",
    ),
    log_artifacts: list[Path] | None = Option(
        None,
        "--log-artifacts",
        "-A",
        help="Paths to files to upload as run artifacts. Repeat the option to add more.",
    ),
    ci: bool = Option(
        False,
        help="Whether to run in CI mode. "
        "If true, it will print the criteria at the end of each epoch instead of using a progress bar.",
    ),
    dist_backend: str | None = Option(
        None,
        envvar="DIST_BACKEND",
        help='Communication library for the distributed process group (e.g. "nccl", "gloo"). '
        "Selected automatically from the device when not specified.",
    ),
    dist_url: str | None = Option(
        None,
        envvar="DIST_URL",
        help='Rendezvous URL for setting up distributed training. Defaults to "env://" when not specified.',
    ),
    strategy_pattern: Any | None = Option(
        None,
        "--strategy",
        parser=path_or_any_parser,
        help=object_pattern_help("the distributed strategy factory", "MyStrategy", call=False)
        + PATH_FORM_HELP
        + " The factory is called with the resolved device and local rank. Defaults to "
        "DistributedDataParallelStrategy when a distributed environment is detected, otherwise SingleDeviceStrategy.",
    ),
) -> None:
    """Train PyTorch models with a Learner, recording the run to an experiment-tracking service."""
    if not model_patterns:
        raise ValueError("At least one model pattern must be provided.")
    device, global_rank, local_rank, world_size, distributed = scm_torch.distributed.initial_distributed_env(
        device=device, dist_backend=dist_backend, dist_url=dist_url, return_dict=False
    )
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision(matmul_precision)
    torch.manual_seed(seed + global_rank)
    np.random.seed(seed + global_rank)
    random.seed(seed + global_rank)
    strategy = _resolve_strategy(strategy_pattern, device, local_rank, distributed)
    is_main = global_rank == 0
    input_shapes = reduce_dict(shapes)
    training_dataset = instantiate_object(training_dataset_pattern)
    validation_dataset = instantiate_object(validation_dataset_pattern) if validation_dataset_pattern else None
    provider = SimpleDataProvider(training_dataset=training_dataset, validation_dataset=validation_dataset)
    if is_main:
        print("Count the dataset sizes...")
        print(f"Training dataset size: {provider.steps_per_epoch} steps.")
        print(f"Validation dataset size: {provider.validation_steps} steps.")
    models, learner, learner_outputs, tracker = _assemble_learner(
        model_patterns=model_patterns,
        input_shapes=input_shapes,
        initializers=instantiator.instantiate(reduce_dict(initializer_patterns)),
        resume=resume,
        strategy=strategy,
        compile_kw=instantiator.instantiate(compile_pattern),
        learner_pattern=learner_pattern,
        learner_outputs=learner_outputs,
        device=device,
        distributed=distributed,
        is_main=is_main,
    )
    # Built before the resume, which fetches the state through it. Only the experiment name is stored
    # here: the run itself starts in __enter__.
    if is_main:
        logger_type = scm_loggers.mlflow.MLflowLogger if logger_name == "mlflow" else scm_loggers.wandb.WandbLogger
        logger: scm_loggers.base.Logger = logger_type(experiment=experiment)
    else:
        logger = scm_loggers.base.NullLogger()
    if resume is not None:
        start_epoch = _restore_training_state(
            resume=resume,
            strategy=strategy,
            models=models,
            learner=learner,
            start_epoch=start_epoch,
            is_main=is_main,
            logger=logger,
        )
    trainer_type = scm_torch.trainer.TorchTrainer if trainer_pattern is None else instantiate_object(trainer_pattern)
    trainer = trainer_type(device=device, learner=learner, tracker=tracker, data=provider, callbacks=[])
    _build_callbacks(
        trainer=trainer,
        provider=provider,
        strategy=strategy,
        learner_outputs=learner_outputs,
        higher_criteria=higher_criteria,
        lower_criteria=lower_criteria,
        save_criteria=save_criteria,
        logger=logger,
        ci=ci,
        is_main=is_main,
    )
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
