"""Flax related commands for the StructCast Model CLI application."""

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, Literal, cast

from structcast.utils.base import dump_yaml_to_string
from typer import Argument, Option, Typer

# A package shim routing to lazy submodules, so importing it pulls in no framework. Wrapping it in
# `LazyModuleImporter` would not work: it copies the shim's still unresolved submodule slots, so
# every access after the first would hand back `None`.
from structcast_model import loggers as scm_loggers
from structcast_model.base_trainer import (
    EVENTS,
    DatasetLike,
    Printer,
    ProgressBar,
    SimpleDataProvider,
    get_dataset,
    get_dataset_size,
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
    warmup_runs,
)
from structcast_model.commands.utils import (
    bool_or_path_or_dict_parser,
    dict_parser,
    get_module_outputs,
    instantiate_object,
    path_or_any_parser,
    reduce_dict,
)

if TYPE_CHECKING:
    import jax
    from structcast.core import instantiator

    from flax import nnx
    from structcast_model.builders import flax as flax_builder
    from structcast_model.flax import distributed as flax_distributed, trainer as flax_trainer
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    jax = LazyModuleImporter("jax")
    instantiator = LazyModuleImporter("structcast.core.instantiator")
    nnx = LazyModuleImporter("flax.nnx")
    flax_builder = LazyModuleImporter("structcast_model.builders.flax")
    flax_distributed = LazyModuleImporter("structcast_model.flax.distributed")
    flax_trainer = LazyModuleImporter("structcast_model.flax.trainer")


app = Typer(no_args_is_help=True)
creator = Typer(no_args_is_help=True)
app.add_typer(creator, name="create", help="Commands for creating Flax nnx modules.")

template_param = template_param_option('For example: --parameter "model: {input_size: 128, output_size: 10}"')
shapes = shapes_option(
    shapes_help('"image: [224, 224, 3]"', "jax.numpy.zeros")
    + " Omit it only when the model declares INPUT_SHAPES itself."
)
device = Option(
    None,
    "--device",
    "-d",
    help='Device the dummy inputs are placed on, named "<platform>:<index>" as listed by JAX (e.g. "cpu:0", '
    '"gpu:0"). If not specified, the first available JAX device is used.',
)
compile_pattern: dict[str, Any] | None = compile_option("nnx.jit")


@creator.command(name="model")
def create_model(
    cfg_path: str = Argument(..., help="Path to the model configuration file."),
    output: str | None = output_script_path,
    parameters: list[dict] | None = template_param,
    classname: str = Option("Model", "--classname", "-n", help="Name of the generated model class."),
    structured_output: bool = Option(
        True,
        help="Return the module outputs as a dict keyed by output name instead of positionally. Defaults to true, "
        "which overrides the configuration's STRUCTURED_OUTPUT; pass --no-structured-output to force plain output. "
        "Ignored with --sublayer: the selected layer's own configuration decides.",
    ),
    sublayer: str | None = Option(
        None, "--sublayer", help="The reference to a sublayer in the template to build instead of the root layer."
    ),
) -> None:
    """Create a Flax nnx module from the given configuration file and parameters."""
    flax_builder.FlaxBuilder.from_path(cfg_path)(
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
    """Create a Flax (nnx) learner class from the given configuration file and parameters."""
    builder = flax_builder.FlaxLearnerBuilder.from_path(cfg_path)
    builder(parameters=reduce_dict(parameters), classname=classname)(output)


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
    warmup_runs: int = warmup_runs,
    times: int = times,
    batch_size: int = batch_size,
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


MATMUL_PRECISIONS: Mapping[str, str] = {"highest": "highest", "high": "high", "medium": "bfloat16"}
"""JAX spelling of the shared precision names; its lowest float32 setting is named after the dtype."""

TRAINING_COMPILE_KW: Mapping[str, Any] = {
    "static_argnames": "need_update",
    "donate_argnames": ("models", "optimizers", "acc_grads"),
}
"""The compilation arguments the generated training step's contract fixes (see `docs/adr/0013`).

`need_update` is a Python bool selecting between the accumulating and the updating variant, so it is
static; the models, the optimizers and the gradient accumulator are rewritten in place every step, so
their buffers are donated. The batch is never donated, and the inference step -- which runs against
views sharing the models' arrays -- donates nothing.
"""


def _compile_parser(value: str) -> dict[str, Any] | None:
    """Parse `--compile`, where "none" and "false" disable compilation.

    Args:
        value (str): The raw option value; empty when the option is left at its default.

    Returns:
        dict[str, Any] | None: The `nnx.jit` keyword arguments to compile with, or None to skip
            compilation.
    """
    if value.strip().lower() in ("none", "false"):
        return None
    return bool_or_path_or_dict_parser(value) or {}


def _config_hash(model_patterns: list[dict], learner_pattern: Any, shapes: Mapping[str, Any]) -> str:
    """Return the digest of what a run trains: its model patterns, its learner pattern and its shapes.

    Recorded in the saved training state so a resumed run can be told apart from the configuration it
    was saved from. The optimizers are not part of it: they are built inside the generated learner
    class, which carries neither its configuration nor a hash of it (issue #22).
    """
    payload = {"models": model_patterns, "learner": learner_pattern, "shapes": shapes}
    return sha256(dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _resolve_strategy(strategy: Any, device: str | None) -> "flax_distributed.FlaxDistributedStrategy":
    """Resolve `--strategy`: a preset name builds the strategy, a pattern builds whatever it names."""
    if isinstance(strategy, str):
        # Cast, not validate: the strategy owns the list of presets it knows and rejects the rest
        # with the names it accepts, which is the error a mistyped preset should read.
        preset = cast('Literal["single", "dp", "fsdp"]', strategy)
        return flax_distributed.FlaxDistributedStrategy(preset=preset, device=device)
    return instantiate_object(strategy)(device=device)


def _strategy_parser(value: str) -> Any:
    """Parse `--strategy`: a bare name is a preset, anything else an object pattern or a path to one."""
    return value if value.isidentifier() else path_or_any_parser(value)


@dataclass(frozen=True)
class _ShardedDataset:
    """A dataset whose batches are placed across the strategy's mesh as they are read.

    Built once per run and handed to the data provider, whose properties must return the same object
    on every read. Counting an epoch goes to the wrapped dataset, so it places nothing.
    """

    dataset: DatasetLike | Callable[[], DatasetLike]
    """The dataset producing the batches."""

    strategy: Any
    """The strategy whose mesh the batches are placed on."""

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield every batch of one epoch, placed on the mesh."""
        return (self.strategy.shard_batch(batch) for batch in get_dataset(self.dataset))

    def __len__(self) -> int:
        """Return the number of batches in one epoch."""
        return get_dataset_size(self.dataset)

    def __post_init__(self) -> None:
        """Take over the wrapped dataset's lifecycle hooks, so the trainer still routes them.

        Copied onto the instance rather than forwarded from `__getattr__`: the trainer selects the
        participants of an event with `isinstance` against a protocol, and a protocol check looks its
        attributes up statically, which never reaches a `__getattr__`.
        """
        for event in EVENTS:
            hook = getattr(self.dataset, event, None)
            if hook is not None:
                object.__setattr__(self, event, hook)


@app.command()
def train(  # noqa: PLR0913, PLR0917  # The CLI surface: every training option is one Typer parameter.
    model_patterns: list[dict] = Argument(
        parser=dict_parser,
        help=object_pattern_help("the model factory", "MyModel", keyed=True, call=False)
        + ' Pass one positional argument per model, each a mapping with exactly one "name: pattern" entry given '
        "inline; a file path is not accepted here. Each pattern resolves to a factory, not to a model: the command "
        'calls it with the run\'s "flax.nnx.Rngs" as rngs=..., built from --seed, which is how a generated model '
        "class is initialized. The names are passed to the --learner factory as keyword arguments.",
    ),
    shapes: list[dict] | None = shapes_option(
        shapes_help('"image: [224, 224, 3]"', "jax.numpy.zeros")
        + " Repeat the option to declare more inputs; occurrences are merged at the top level, so an input named "
        "twice keeps only the last occurrence. When omitted, the INPUT_SHAPES declared by the built models are "
        "used, merged across them. Nothing is allocated from them here -- Flax modules build their parameters in "
        "their constructor -- so they are recorded with the run and identify its configuration.",
    ),
    device: str | None = Option(
        None,
        "--device",
        "-d",
        help='Device the "single" strategy runs on, named "<platform>:<index>" as listed by JAX (e.g. "cpu:0", '
        '"gpu:0"). If not specified, the first available JAX device is used. The multi-device strategies span the '
        "devices themselves and ignore it.",
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
    compile_pattern: dict[str, Any] | None = Option(
        "",
        "--compile",
        "-c",
        parser=_compile_parser,
        show_default="nnx.jit",
        help='Whether to compile the Learner\'s steps with "flax.nnx.jit", which is what a Flax run is expected to '
        'do and therefore the default. Pass "none" or "false" to run them eagerly, or a dictionary of extra '
        '"nnx.jit" keyword arguments -- or a path to a YAML/JSON file holding one -- to add to the defaults. The '
        "arguments deciding what is static and what is donated are the generated step's contract and cannot be "
        "overridden.",
    ),
    trainer_pattern: Any | None = Option(
        None,
        "--trainer",
        parser=path_or_any_parser,
        help=object_pattern_help("the trainer", "MyTrainer", call=False)
        + PATH_FORM_HELP
        + " The pattern must build a callable accepted as trainer(learner=..., tracker=..., data=..., "
        "callbacks=[]) whose result exposes training_prefix, validation_prefix, callbacks, describe() and "
        "fit(epochs, start_epoch, validation_frequency); subclassing FlaxTrainer is the intended way.",
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
        "supported. Restores models and optimizers, and continues from the saved epoch. The models are built "
        "exactly as they are without it -- the restore overwrites the initialization rather than replacing it. "
        "Resume is exact only at epoch boundaries: the saved state carries the epoch, step and update counters "
        "but no data-order or RNG state, so the resumed epoch restarts from the beginning of the dataset.",
    ),
    training_dataset_pattern: Any = Option(
        ...,
        "--training-dataset",
        parser=path_or_any_parser,
        help=object_pattern_help("the training dataset", "MyDataset")
        + PATH_FORM_HELP
        + " Every batch it yields is placed across the strategy's mesh before it reaches the Learner, so each "
        "entry needs a leading dimension the mesh size divides.",
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
        'prefix (e.g. "val_accuracy"). A name no epoch produces is silently never monitored, and omitting the '
        "option monitors nothing.",
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
    seed: int = Option(
        42,
        envvar="SEED",
        help="Random seed for reproducibility. It keys the run's RNG streams, so it decides both the model "
        "initialization and every random draw a step makes.",
    ),
    matmul_precision: Literal["highest", "high", "medium"] = Option(
        "high",
        envvar="MATMUL_PRECISION",
        help='Precision for float32 matrix multiplications: "highest" keeps full float32, while "high" and '
        '"medium" trade accuracy for tensor-core speed, the latter by computing in bfloat16.',
    ),
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
    strategy_pattern: Any = Option(
        "single",
        "--strategy",
        parser=_strategy_parser,
        help='How the run uses the devices: the preset name "single" (one device), "dp" (the batch split across '
        'every device, parameters replicated) or "fsdp" (the batch split and the parameters sharded too). '
        + object_pattern_help("a strategy factory", "MyStrategy", call=False).replace(
            "The object pattern", "Or the object pattern"
        )
        + PATH_FORM_HELP
        + " The factory is called with the resolved device; the templates under cfg/flax/strategies bind the "
        "remaining knobs.",
    ),
) -> None:
    """Train Flax (nnx) models with a Learner, recording the run to an experiment-tracking service."""
    if not model_patterns:
        raise ValueError("At least one model pattern must be provided.")
    jax.config.update("jax_default_matmul_precision", MATMUL_PRECISIONS[matmul_precision])
    # First: constructing the strategy activates its mesh process-wide, and every array allocated
    # afterwards -- the model parameters above all -- is placed against it.
    strategy = _resolve_strategy(strategy_pattern, device)
    rngs = nnx.Rngs(params=jax.random.key(seed), dropout=jax.random.fold_in(jax.random.key(seed), 1))
    models: OrderedDict[str, Any] = OrderedDict()
    for raw in model_patterns:
        if len(raw) != 1:
            raise ValueError(f"Each model pattern should contain exactly one model definition. Got: {raw}")
        model_name, pattern = next(iter(raw.items()))
        factory = instantiate_object(pattern)
        if isinstance(factory, nnx.Module):
            # The pattern already called the class, so the command holds a built module and the run's
            # RNG never reached it. Calling it anyway would run the module's forward pass with a
            # `rngs` keyword and fail somewhere inside the generated model instead.
            raise ValueError(
                f'The pattern of model "{model_name}" builds the module itself, so the run\'s seeded RNG cannot '
                'reach it. Drop the "_call_" entry and let the command call the class with rngs.'
            )
        models[model_name] = factory(rngs=rngs)
    input_shapes = flax_trainer.resolve_input_shapes(models, reduce_dict(shapes)) or {}
    config_hash = _config_hash(model_patterns, learner_pattern, input_shapes)
    # Before the learner: an optimizer inherits the sharding of the parameters it is built over, and
    # the learner's inference views are taken from the models as they are when it is constructed.
    models = strategy.wrap(models)
    learner = instantiate_object(learner_pattern)(**models)
    outputs = get_module_outputs(learner, learner_outputs, "learner")
    compile_kw = None if compile_pattern is None else instantiator.instantiate(compile_pattern)
    if compile_kw is not None:
        # Both spellings of both contract arguments go: --help promises they cannot be overridden,
        # and the argnums form would renumber what the argnames form fixes.
        fixed = {"static_argnames", "static_argnums", "donate_argnames", "donate_argnums"}
        extra = {key: value for key, value in compile_kw.items() if key not in fixed}
        # `flow_functions` is what a generated learner exposes, not part of the Learner protocol:
        # a hand-written learner has nothing to bind, as in `cmd_torch`.
        for flow_name in list(getattr(learner, "flow_functions", ())):
            # The generated learner names its steps after the contract they follow: only the training
            # one takes the static flag and the donated state (`docs/adr/0015`).
            arguments = {**TRAINING_COMPILE_KW, **extra} if flow_name == "_training_step" else extra
            setattr(learner, flow_name, strategy.compile(getattr(learner, flow_name), arguments))
    provider = SimpleDataProvider(
        training_dataset=_ShardedDataset(instantiate_object(training_dataset_pattern), strategy),
        validation_dataset=(
            None
            if validation_dataset_pattern is None
            else _ShardedDataset(instantiate_object(validation_dataset_pattern), strategy)
        ),
    )
    print("Count the dataset sizes...")
    print(f"Training dataset size: {provider.steps_per_epoch} steps.")
    print(f"Validation dataset size: {provider.validation_steps} steps.")
    # Built before the resume, which fetches the state through it. Only the experiment name is stored
    # here: the run itself starts in __enter__.
    logger_type = scm_loggers.mlflow.MLflowLogger if logger_name == "mlflow" else scm_loggers.wandb.WandbLogger
    logger: scm_loggers.base.Logger = logger_type(
        experiment=experiment, state_backend=scm_loggers.state_backends.FlaxStateBackend()
    )
    if resume is not None:
        start_epoch = flax_trainer.restore_training_state(
            resume=resume,
            strategy=strategy,
            models=models,
            learner=learner,
            start_epoch=start_epoch,
            logger=logger,
            config_hash=config_hash,
        )
    trainer_type = flax_trainer.FlaxTrainer if trainer_pattern is None else instantiate_object(trainer_pattern)
    trainer = trainer_type(
        learner=learner, tracker=flax_trainer.FlaxTracker.from_criteria(outputs), data=provider, callbacks=[]
    )
    display = (
        Printer()
        if ci
        else ProgressBar(
            steps_per_epoch=provider.steps_per_epoch,
            validation_steps=provider.validation_steps,
            training_criteria=[f"{trainer.training_prefix}{name}" for name in outputs],
            validation_criteria=[f"{trainer.validation_prefix}{name}" for name in outputs],
        )
    )
    saver = flax_trainer.FlaxTrainingStateSaver(
        logger=logger,
        strategy=strategy,
        # The optimizer hashes stay empty: they would have to come from the OPTIMIZER patterns, which
        # the generated learner class does not carry (issue #22). The resume check skips what is
        # missing, so the slot is here and unfilled rather than absent.
        extra_meta={"seed": seed, "config_hash": config_hash, "optimizer_hashes": {}},
    )
    bests = flax_trainer.FlaxBestCriterion.from_criteria(
        higher_criteria, lower_criteria, save_criteria, logger=logger, strategy=strategy
    )
    trainer.callbacks = [display, logger, saver, *bests]
    arguments = {
        **reduce_dict(log_arguments),
        "models": model_patterns,
        "parameters": {
            name: sum(leaf.size for leaf in jax.tree.leaves(nnx.state(model, nnx.Param)))
            for name, model in models.items()
        },
        "shapes": input_shapes,
        "device": device,
        "strategy": strategy_pattern,
        # A plain dict: the mesh reports its shape as an OrderedDict, which YAML tags as Python.
        "mesh": dict(strategy.mesh.shape),
        "learner": learner_pattern,
        "learner_outputs": outputs,
        "compile": compile_pattern,
        "trainer": trainer_pattern,
        "epochs": epochs,
        "start_epoch": start_epoch,
        "resume": resume,
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
    with logger:
        logger.log_params(
            {
                "jax_version": jax.__version__,
                "jax_backend": jax.default_backend(),
                "epochs": epochs,
                "steps_per_epoch": provider.steps_per_epoch,
                "validation_steps": provider.validation_steps,
            }
        )
        logger.log_dict(arguments, "arguments.yaml")
        for artifact in log_artifacts or []:
            logger.log_artifact(str(artifact))
        print(f"Registered callbacks:\n{dump_yaml_to_string(trainer.describe())}")
        trainer.fit(epochs=epochs, start_epoch=start_epoch, validation_frequency=validation_frequency)


__all__ = ["app"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
