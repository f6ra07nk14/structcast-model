"""Keras related commands for the StructCast Model CLI application."""

from collections import OrderedDict
from collections.abc import Mapping
from hashlib import sha256
import json
from json import dumps
import os
from pathlib import Path
import sys
from time import time
from typing import TYPE_CHECKING, Any, Literal, cast

from structcast.utils.base import dump_yaml_to_string
from typer import Argument, Option, Typer

# A package shim routing to lazy submodules, so importing it pulls in no framework, as in cmd_flax.
from structcast_model import loggers as scm_loggers
from structcast_model.base_trainer import Printer, ProgressBar, SimpleDataProvider
from structcast_model.commands.shared_args import (
    PATH_FORM_HELP,
    batch_size,
    ci,
    epochs,
    experiment,
    higher_criteria,
    learner_outputs,
    learner_pattern,
    log_arguments,
    log_artifacts,
    logger_name,
    lower_criteria,
    model_pattern,
    object_pattern_help,
    output_script_path,
    resume_option,
    save_criteria,
    seed_option,
    shapes_help,
    shapes_option,
    start_epoch,
    template_param_option,
    times,
    trainer_option,
    training_dataset_option,
    training_mode,
    validation_dataset_pattern,
    validation_frequency,
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

    import keras
    from structcast_model.builders import keras as keras_builder
    from structcast_model.keras import distributed as keras_distributed, trainer as keras_trainer
    import torch
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    jax = LazyModuleImporter("jax")
    instantiator = LazyModuleImporter("structcast.core.instantiator")
    keras = LazyModuleImporter("keras")
    keras_builder = LazyModuleImporter("structcast_model.builders.keras")
    keras_distributed = LazyModuleImporter("structcast_model.keras.distributed")
    keras_trainer = LazyModuleImporter("structcast_model.keras.trainer")
    torch = LazyModuleImporter("torch")


app = Typer(no_args_is_help=True)
creator = Typer(no_args_is_help=True)
app.add_typer(creator, name="create", help="Commands for creating Keras layer classes.")

template_param = template_param_option('For example: --parameter "model: {input_size: 128, output_size: 10}"')
shapes = shapes_option(
    shapes_help('"image: [224, 224, 3]"', "numpy.zeros") + " Omit it only when the model declares INPUT_SHAPES itself."
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
train_device = Option(
    None,
    "--device",
    "-d",
    help='Device the run is checked against, named as returned by "keras.distribution.list_devices()", e.g. '
    '"cpu:0" or "gpu:0"; an unavailable name aborts and the first listed device is used when omitted. It places '
    "nothing: which devices a Keras backend computes on is the backend's own choice (restrict it with "
    "CUDA_VISIBLE_DEVICES), so the name is validated and recorded with the run. Spanning several devices is a "
    "distributed strategy, which this command does not have yet.",
)
backend_option = Option(
    ...,
    "--backend",
    envvar="KERAS_BACKEND",
    help="Keras backend the run executes on. Keras resolves it once, while it is first imported, so the command "
    "sets it before importing Keras and fails when a different one is already live. There is no default: the "
    "backend decides what a run computes on, so it is stated rather than inherited from ~/.keras/keras.json.",
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


@creator.command(name="learner")
def create_learner(
    cfg_path: str = Argument(..., help="Path to the learner configuration file."),
    output: str | None = output_script_path,
    parameters: list[dict] | None = template_param,
    classname: str = Option("Learner", "--classname", "-n", help="Name of the generated Learner class."),
) -> None:
    """Create a Keras learner class from the given configuration file and parameters.

    The generated learner is backend-neutral -- it names no backend and imports no framework beyond
    `keras` -- and this command only writes it, so it takes no --backend.
    """
    builder = keras_builder.KerasLearnerBuilder.from_path(cfg_path)
    builder(parameters=reduce_dict(parameters), classname=classname)(output)


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
    # Unlike `train`, this command takes no --backend and inherits the ambient one (`docs/adr/0016`),
    # so it says which one produced the number: the backend decides what actually executes.
    print(f'Timing on the "{keras.backend.backend()}" Keras backend, device "{device}".')
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


def _activate_backend(backend: str) -> None:
    """Bind the Keras backend for this process, before anything imports Keras.

    Keras reads `KERAS_BACKEND` once, while it is first imported, and never switches afterwards: a
    command that set the variable after the import would run on the old backend while reporting the
    requested one. Both checks exist because there are two ways to get there -- Keras already
    imported by the caller, and Keras importing something else than the variable asked for.
    """
    imported = sys.modules.get("keras")
    if imported is not None and (active := imported.backend.backend()) != backend:
        raise ValueError(
            f'Keras is already running on the "{active}" backend, so --backend {backend} cannot take effect: '
            "the backend is resolved once, when Keras is first imported. Run the command in a fresh process."
        )
    os.environ["KERAS_BACKEND"] = backend
    # The first attribute access is what imports Keras, hence after the assignment above.
    if (active := keras.backend.backend()) != backend:
        raise ValueError(f'Keras started on the "{active}" backend instead of the requested "{backend}".')


def _mixed_precision_policy(factory: Any) -> str | None:
    """Return the global policy the generated learner's module asks for, or None for a plain run.

    A generated learner class is loaded from a file rather than imported by name, so its module
    never lands in `sys.modules`: the globals its methods were defined in are the only handle on the
    constants next to it, as in `cmd_flax._optimizer_hashes`. A hand-written learner declares none
    and gets no policy.
    """
    namespace = getattr(getattr(factory, "__init__", None), "__globals__", {})
    raw = namespace.get("__mixed_precision__")
    # The predicate of `keras.adapters.prepare`: any mapping enables the policy, an empty one
    # included, so that a run is not loss-scaled by the adapter while computing in float32.
    enabled = raw if isinstance(raw, bool) else isinstance(raw, Mapping)
    if not enabled:
        return None
    return f"mixed_{namespace['__mixed_precision_type__']}"


def _config_hash(model_patterns: list[dict], learner_pattern: Any, shapes: Mapping[str, Any]) -> str:
    """Return the digest of what a run trains: its model patterns, its learner pattern and its shapes.

    Recorded in the saved training state so a resumed run can be told apart from the configuration it
    was saved from, exactly as `cmd_flax` records it. The optimizers are not part of it: they are
    hashed separately, by the builder that emits them, and reported per segment.
    """
    payload = {"models": model_patterns, "learner": learner_pattern, "shapes": shapes}
    return sha256(dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _optimizer_hashes(learner: Any) -> Mapping[str, str]:
    """Return the `__optimizer_hashes__` the learner's own module declares, empty for anything else.

    Read off the globals the learner's methods were defined in, as `_mixed_precision_policy` reads
    the mixed precision constants: a generated learner is loaded from a file, so its module never
    lands in `sys.modules`. A hand-written learner declares none, and the resume check skips what is
    missing.
    """
    namespace = getattr(type(learner).__init__, "__globals__", {})
    return cast(Mapping[str, str], namespace.get("__optimizer_hashes__") or {})


def _resolve_strategy(strategy: Any, device: str | None) -> "keras_distributed.KerasDistributedStrategy":
    """Resolve `--strategy`: a preset name builds the strategy, a pattern builds whatever it names."""
    if isinstance(strategy, str):
        # Cast, not validate: the strategy owns the list of presets it knows, and which of them the
        # active backend supports, and rejects the rest with the reason -- which is the error to read.
        preset = cast('Literal["single", "dp", "fsdp"]', strategy)
        return keras_distributed.KerasDistributedStrategy(preset=preset, device=device)
    return instantiate_object(strategy)(device=device)


def _strategy_parser(value: str) -> Any:
    """Parse `--strategy`: a bare name is a preset, anything else an object pattern or a path to one."""
    return value if value.isidentifier() else path_or_any_parser(value)


@app.command()
def train(  # noqa: PLR0913, PLR0917  # The CLI surface: every training option is one Typer parameter.
    model_patterns: list[dict] = Argument(
        parser=dict_parser,
        help=object_pattern_help("the model", "MyModel", keyed=True)
        + ' Pass one positional argument per model, each a mapping with exactly one "name: pattern" entry given '
        "inline; a file path is not accepted here. A pattern building a layer is traced into a model with the "
        "run's shapes before the Learner is built, because a Keras layer owns no variables until it is. The names "
        "are passed to the --learner factory as keyword arguments.",
    ),
    backend: Literal["tensorflow", "jax", "torch"] = backend_option,
    shapes: list[dict] | None = shapes_option(
        shapes_help('"image: [224, 224, 3]"', "numpy.zeros")
        + " Repeat the option to declare more inputs; occurrences are merged at the top level, so an input named "
        "twice keeps only the last occurrence. When omitted, each model is traced with the INPUT_SHAPES it "
        "declares itself; given, the same shapes are used for every model."
    ),
    device: str | None = train_device,
    learner_pattern: Any = learner_pattern,
    learner_outputs: list[str] | None = learner_outputs,
    trainer_pattern: Any | None = trainer_option(
        "trainer(learner=..., tracker=..., data=..., callbacks=[])", "KerasTrainer"
    ),
    epochs: int = epochs,
    start_epoch: int = start_epoch,
    resume: str | None = resume_option(
        "Restores models and optimizers, and continues from the saved epoch. The state carries the Keras backend "
        "it was written on and a resume onto another one is refused: normalization statistics and RNG "
        "trajectories are not verified equivalent across backends.",
        "data-order or RNG",
    ),
    training_dataset_pattern: Any = training_dataset_option(
        " Every batch it yields is passed to the Learner as it is; a multi-device strategy places it across the "
        "replicas first, so each entry needs a leading dimension the replica count divides -- except on the "
        "torch backend, where the loader hands each rank its own slice."
    ),
    validation_dataset_pattern: Any | None = validation_dataset_pattern,
    validation_frequency: int = validation_frequency,
    lower_criteria: list[str] = lower_criteria,
    higher_criteria: list[str] = higher_criteria,
    save_criteria: list[str] = save_criteria,
    seed: int = seed_option(
        ' It is applied with "keras.utils.set_random_seed" before the models are built, which seeds Python, NumPy '
        "and the active backend, so it decides both the model initialization and every random draw a step makes."
    ),
    experiment: str = experiment,
    logger_name: Literal["mlflow", "wandb"] = logger_name,
    log_arguments: list[dict] | None = log_arguments,
    log_artifacts: list[Path] | None = log_artifacts,
    ci: bool = ci,
    strategy_pattern: Any = Option(
        "single",
        "--strategy",
        parser=_strategy_parser,
        help='How the run uses the devices: the preset name "single" (one device), "dp" (the batch split across '
        'the replicas, variables replicated) or "fsdp" (the batch split and the variables sharded too). Each '
        "preset runs on the mechanism its Keras backend actually has -- keras.distribution on jax, "
        "tf.distribute.MirroredStrategy on tensorflow, DistributedDataParallel on torch -- and fsdp is available "
        "on jax alone; the other two cells are refused with the reason. "
        + object_pattern_help("a strategy factory", "MyStrategy", call=False, lead="Or the object pattern")
        + PATH_FORM_HELP
        + " The factory is called with the resolved device; the templates under cfg/keras/strategies bind the "
        "remaining knobs.",
    ),
) -> None:
    """Train Keras models with a Learner, recording the run to an experiment-tracking service."""
    if not model_patterns:
        raise ValueError("At least one model pattern must be provided.")
    _activate_backend(backend)
    device = keras_trainer.get_keras_device(device)
    keras.utils.set_random_seed(seed)
    # Resolved before the models: the class itself is what carries the policy the models are built
    # under, and instantiating it needs them.
    factory = instantiate_object(learner_pattern)
    if (policy := _mixed_precision_policy(factory)) is not None:
        # A policy only reaches the layers built after it is set, and the learner receives models
        # that are already built, so this is the last moment it can be set (`docs/adr/0016`).
        print(f'Setting the global mixed precision policy to "{policy}"...')
        keras.mixed_precision.set_global_policy(policy)
    strategy = _resolve_strategy(strategy_pattern, device)
    models: OrderedDict[str, Any] = OrderedDict()
    declared: dict[str, Any] = {}
    # Everything a run allocates is built inside the activation: a JAX variable reads the active
    # distribution while it is created, and a MirroredStrategy mirrors only what its scope encloses
    # -- the models above all, and the optimizers the learner builds against their variables.
    with strategy.activate():
        for raw in model_patterns:
            if len(raw) != 1:
                raise ValueError(f"Each model pattern should contain exactly one model definition. Got: {raw}")
            model_name, pattern = next(iter(raw.items()))
            built = instantiate_object(pattern)
            # Read before the trace: `initial_model` wraps a layer into a functional `keras.Model`,
            # which carries none of the layer's attributes, so the shapes it was traced with would be
            # unrecoverable afterwards and the run would record none.
            declared.update(keras_trainer.resolve_input_shapes(built) or {})
            models[model_name] = keras_trainer.initial_model(built, reduce_dict(shapes))
        strategy.sync_initial_weights(models)
        # Before the learner: it captures the model objects it is handed, and its optimizers are
        # built against their variables while it is constructed.
        models = strategy.wrap(models)
        learner = factory(**models)
    # A declared shape is a tuple, which `arguments.yaml` would record as a `!!python/tuple` tag no
    # safe YAML loader reads back; the round-trip makes it the plain data `--shape` would have given.
    input_shapes = reduce_dict(shapes) or json.loads(json.dumps(declared))
    config_hash = _config_hash(model_patterns, learner_pattern, input_shapes)
    # After the learner: the steps this rewires are the ones the backend adapter built in its
    # constructor, and each one runs the replicas itself rather than being traced into a scope.
    strategy.wrap_steps(learner)
    outputs = get_module_outputs(learner, learner_outputs, "learner")
    provider = SimpleDataProvider(
        training_dataset=instantiate_object(training_dataset_pattern),
        validation_dataset=(
            None if validation_dataset_pattern is None else instantiate_object(validation_dataset_pattern)
        ),
    )
    print("Count the dataset sizes...")
    print(f"Training dataset size: {provider.steps_per_epoch} steps.")
    print(f"Validation dataset size: {provider.validation_steps} steps.")
    # Only the experiment name is stored here: the run itself starts in __enter__.
    logger_type = scm_loggers.mlflow.MLflowLogger if logger_name == "mlflow" else scm_loggers.wandb.WandbLogger
    logger: scm_loggers.base.Logger = logger_type(
        experiment=experiment, state_backend=scm_loggers.state_backends.KerasStateBackend()
    )
    optimizer_hashes = _optimizer_hashes(learner)
    if resume is not None:
        start_epoch = keras_trainer.restore_training_state(
            resume=resume,
            strategy=strategy,
            models=models,
            learner=learner,
            start_epoch=start_epoch,
            logger=logger,
            optimizer_hashes=optimizer_hashes,
            config_hash=config_hash,
            is_main=strategy.is_main,
        )
    trainer_type = keras_trainer.KerasTrainer if trainer_pattern is None else instantiate_object(trainer_pattern)
    trainer = trainer_type(
        learner=learner, tracker=keras_trainer.KerasTracker.from_criteria(outputs), data=provider, callbacks=[]
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
    saver = keras_trainer.KerasTrainingStateSaver(
        logger=logger,
        strategy=strategy,
        extra_meta={"seed": seed, "config_hash": config_hash, "optimizer_hashes": dict(optimizer_hashes)},
    )
    bests = keras_trainer.KerasBestCriterion.from_criteria(
        higher_criteria, lower_criteria, save_criteria, logger=logger, strategy=strategy
    )
    trainer.callbacks = [display, logger, saver, *bests]
    arguments = {
        **reduce_dict(log_arguments),
        "models": model_patterns,
        "parameters": {name: model.count_params() for name, model in models.items()},
        "shapes": input_shapes,
        "backend": backend,
        "device": device,
        "strategy": strategy_pattern,
        "replicas": strategy.replicas,
        "mixed_precision_policy": policy,
        "learner": learner_pattern,
        "learner_outputs": outputs,
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
        "experiment": experiment,
        "logger": logger_name,
        "ci": ci,
    }
    with logger:
        logger.log_params(
            {
                "keras_version": keras.__version__,
                "keras_backend": backend,
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
