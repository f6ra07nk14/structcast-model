"""Flax related commands for the StructCast Model CLI application."""

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, Literal, cast

from structcast.utils.base import dump_yaml_to_string
from typer import Argument, Option, Typer

# `scm`, `scm_flax` and `scm_loggers` are package shims routing to lazy submodules, so importing
# them pulls in no framework. Wrapping them in `LazyModuleImporter` would not work: it copies the
# shim's still unresolved submodule slots, so every access after the first would hand back `None`.
import structcast_model as scm
import structcast_model.commands.shared_args as scm_args
from structcast_model.commands.utils import (
    bool_or_path_or_dict_parser,
    config_hash,
    dict_parser,
    get_module_outputs,
    instantiate_object,
    reduce_dict,
    strategy_parser,
)
import structcast_model.flax as scm_flax
import structcast_model.loggers as scm_loggers

if TYPE_CHECKING:
    import jax
    from structcast.core import instantiator

    from flax import nnx
    from structcast_model.builders import flax as flax_builder
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    jax = LazyModuleImporter("jax")
    instantiator = LazyModuleImporter("structcast.core.instantiator")
    nnx = LazyModuleImporter("flax.nnx")
    flax_builder = LazyModuleImporter("structcast_model.builders.flax")


app = Typer(no_args_is_help=True)
creator = Typer(no_args_is_help=True)
app.add_typer(creator, name="create", help="Commands for creating Flax nnx modules.")

template_param = scm_args.template_param_option('For example: --parameter "model: {input_size: 128, output_size: 10}"')
shapes = scm_args.shapes_option(
    scm_args.shapes_help('"image: [224, 224, 3]"', "jax.numpy.zeros")
    + " Omit it only when the model declares INPUT_SHAPES itself."
)
device = Option(
    None,
    "--device",
    "-d",
    help='Device the dummy inputs are placed on, named "<platform>:<index>" as listed by JAX (e.g. "cpu:0", '
    '"gpu:0"). If not specified, the first available JAX device is used.',
)
compile_pattern: dict[str, Any] | None = scm_args.compile_option("nnx.jit")


@creator.command(name="model")
def create_model(
    cfg_path: str = Argument(..., help="Path to the model configuration file."),
    output: str | None = scm_args.output_script_path,
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
    output: str | None = scm_args.output_script_path,
    parameters: list[dict] | None = template_param,
    classname: str = Option("Learner", "--classname", "-n", help="Name of the generated Learner class."),
) -> None:
    """Create a Flax (nnx) learner class from the given configuration file and parameters."""
    builder = flax_builder.FlaxLearnerBuilder.from_path(cfg_path)
    builder(parameters=reduce_dict(parameters), classname=classname)(output)


@app.command(name="time")
def measure_inference_time(
    model_pattern: Any = scm_args.model_pattern,
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
    warmup_runs: int = scm_args.warmup_runs,
    times: int = scm_args.times,
    batch_size: int = scm_args.batch_size,
) -> None:
    """Measure the average inference time of a Flax model."""
    jax_device = scm_flax.get_jax_device(device)
    training_mode_kw = (
        {"training": training_mode, "deterministic": not training_mode, "use_running_average": not training_mode}
        if training_mode_kwargs_pattern is None
        else instantiator.instantiate(training_mode_kwargs_pattern)
    )
    print("Initializing the model...")
    model = instantiate_object(model_pattern)
    shapes = scm_flax.resolve_input_shapes(model, shapes)
    model = nnx.view(model, raise_if_not_found=False, **training_mode_kw)
    if compile_pattern is None:
        print("Skipping compilation...")
    else:
        print("Compiling the model...")
        model = nnx.jit(model, **instantiator.instantiate(compile_pattern))

    def _measure_single_run() -> float:
        inputs = jax.device_put(scm_flax.create_jax_inputs(shapes, batch_size=batch_size), device=jax_device)
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


def _optimizer_hashes(learner: Any) -> Mapping[str, str]:
    """Return the `OPTIMIZER_HASHES` the learner's own class declares, empty for anything else.

    A generated learner carries the digests as a class attribute (see `docs/adr/0019`), which is the
    only handle on them: the class is loaded from a file rather than imported by name, so its module
    never lands in `sys.modules`. A hand-written learner declares none, and the resume check skips
    what is missing.
    """
    return cast(Mapping[str, str], getattr(type(learner), "OPTIMIZER_HASHES", None) or {})


def _resolve_strategy(strategy: Any, device: str | None) -> "scm_flax.FlaxDistributedStrategy":
    """Resolve `--strategy`: a preset name builds the strategy, a pattern builds whatever it names."""
    if isinstance(strategy, str):
        # Cast, not validate: the strategy owns the list of presets it knows and rejects the rest
        # with the names it accepts, which is the error a mistyped preset should read.
        preset = cast('Literal["single", "dp", "fsdp"]', strategy)
        return scm_flax.FlaxDistributedStrategy(preset=preset, device=device)
    return instantiate_object(strategy)(device=device)


@app.command()
def train(  # noqa: PLR0913, PLR0917  # The CLI surface: every training option is one Typer parameter.
    model_patterns: list[dict] = Argument(
        parser=dict_parser,
        help=scm_args.object_pattern_help("the model factory", "MyModel", keyed=True, call=False)
        + ' Pass one positional argument per model, each a mapping with exactly one "name: pattern" entry given '
        "inline; a file path is not accepted here. Each pattern resolves to a factory, not to a model: the command "
        'calls it with the run\'s "flax.nnx.Rngs" as rngs=..., built from --seed, which is how a generated model '
        "class is initialized. The names are passed to the --learner factory as keyword arguments.",
    ),
    shapes: list[dict] | None = scm_args.shapes_option(
        scm_args.shapes_help('"image: [224, 224, 3]"', "jax.numpy.zeros")
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
    learner_pattern: Any = scm_args.learner_pattern,
    learner_outputs: list[str] | None = scm_args.learner_outputs,
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
    trainer_pattern: Any | None = scm_args.trainer_option(
        "trainer(learner=..., tracker=..., data=..., callbacks=[])", "FlaxTrainer"
    ),
    epochs: int = scm_args.epochs,
    start_epoch: int = scm_args.start_epoch,
    resume: str | None = scm_args.resume_option(
        "Restores models and optimizers, and continues from the saved epoch. The models are built exactly as they "
        "are without it -- the restore overwrites the initialization rather than replacing it.",
        "data-order or RNG",
    ),
    training_dataset_pattern: Any = scm_args.training_dataset_option(
        " Every batch it yields is placed across the strategy's mesh before it reaches the Learner, so each "
        "entry needs a leading dimension the mesh size divides."
    ),
    validation_dataset_pattern: Any | None = scm_args.validation_dataset_pattern,
    validation_frequency: int = scm_args.validation_frequency,
    lower_criteria: list[str] = scm_args.lower_criteria,
    higher_criteria: list[str] = scm_args.higher_criteria,
    save_criteria: list[str] = scm_args.save_criteria,
    seed: int = scm_args.seed_option(
        " It keys the run's RNG streams, so it decides both the model initialization and every random draw a "
        "step makes."
    ),
    matmul_precision: Literal["highest", "high", "medium"] = scm_args.matmul_precision_option(
        ", the latter by computing in bfloat16"
    ),
    experiment: str = scm_args.experiment,
    logger_name: Literal["mlflow", "wandb"] = scm_args.logger_name,
    log_arguments: list[dict] | None = scm_args.log_arguments,
    log_artifacts: list[Path] | None = scm_args.log_artifacts,
    ci: bool = scm_args.ci,
    strategy_pattern: Any = Option(
        "single",
        "--strategy",
        parser=strategy_parser,
        help='How the run uses the devices: the preset name "single" (one device), "dp" (the batch split across '
        'every device, parameters replicated) or "fsdp" (the batch split and the parameters sharded too). '
        + scm_args.object_pattern_help("a strategy factory", "MyStrategy", call=False, lead="Or the object pattern")
        + scm_args.PATH_FORM_HELP
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
    input_shapes = scm_flax.resolve_input_shapes(models, reduce_dict(shapes)) or {}
    config_digest = config_hash(model_patterns, learner_pattern, input_shapes)
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
        for flow_name in list(learner.flow_functions):
            # The generated learner names its steps after the contract they follow: only the training
            # one rewrites state, so only its parameters are donated (`docs/adr/0015`). The inference
            # step runs against views sharing the models' arrays and donates nothing.
            step = getattr(learner, flow_name)
            donated = {"donate_argnames": scm_flax.donate_argnames(step)} if flow_name == "_training_step" else {}
            setattr(learner, flow_name, strategy.compile(step, {**donated, **extra}))
    provider = scm.SimpleDataProvider(
        training_dataset=scm_flax.ShardedDataset(instantiate_object(training_dataset_pattern), strategy),
        validation_dataset=(
            None
            if validation_dataset_pattern is None
            else scm_flax.ShardedDataset(instantiate_object(validation_dataset_pattern), strategy)
        ),
    )
    print("Count the dataset sizes...")
    print(f"Training dataset size: {provider.steps_per_epoch} steps.")
    print(f"Validation dataset size: {provider.validation_steps} steps.")
    # Built before the resume, which fetches the state through it. Only the experiment name is stored
    # here: the run itself starts in __enter__.
    logger_type = scm_loggers.MLflowLogger if logger_name == "mlflow" else scm_loggers.WandbLogger
    logger: scm_loggers.Logger = logger_type(experiment=experiment, state_backend=scm_loggers.FlaxStateBackend())
    optimizer_hashes = _optimizer_hashes(learner)
    if resume is not None:
        start_epoch = scm_flax.restore_training_state(
            resume=resume,
            strategy=strategy,
            models=models,
            learner=learner,
            start_epoch=start_epoch,
            logger=logger,
            optimizer_hashes=optimizer_hashes,
            config_hash=config_digest,
        )
    trainer_type = scm_flax.FlaxTrainer if trainer_pattern is None else instantiate_object(trainer_pattern)
    trainer = trainer_type(
        learner=learner, tracker=scm_flax.FlaxTracker.from_criteria(outputs), data=provider, callbacks=[]
    )
    display = (
        scm.Printer()
        if ci
        else scm.ProgressBar(
            steps_per_epoch=provider.steps_per_epoch,
            validation_steps=provider.validation_steps,
            training_criteria=[f"{trainer.training_prefix}{name}" for name in outputs],
            validation_criteria=[f"{trainer.validation_prefix}{name}" for name in outputs],
        )
    )
    saver = scm_flax.FlaxTrainingStateSaver(
        logger=logger,
        strategy=strategy,
        extra_meta={"seed": seed, "config_hash": config_digest, "optimizer_hashes": dict(optimizer_hashes)},
    )
    bests = scm_flax.FlaxBestCriterion.from_criteria(
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
