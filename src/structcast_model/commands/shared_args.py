"""Typer declarations shared across the commands package.

Where a declaration lives follows three rules (see docs/adr/0010):

- Identical in two or more frameworks: one shared `Option`/`Argument` instance declared here. Typer copies
  the info object per registered parameter, so sharing is safe as long as the parsers stay stateless and
  every signature keeps its own correct annotation.
- Near-identical: a factory taking the differences as explicit arguments, so drift has to get past a
  function signature instead of a copy-paste.
- Framework-specific semantics stay in the owning `cmd_*.py`, next to the code that gives them meaning.

Short flags follow "one letter, one meaning" across the whole app tree; `tests/commands/test_main.py`
enforces it.
"""

from typing import TYPE_CHECKING, Any

from typer import Argument, Option

from structcast_model.commands.utils import (
    bool_or_path_or_dict_parser,
    dict_parser,
    path_or_any_parser,
    tensor_shape_parser,
)

TEMPLATE_PARAM_HELP = (
    "Parameters to format the template configuration file with. "
    'Each parameter is "group: {...}", where the group name selects the parameter group '
    '("SHARED" applies to every group, "DEFAULT" to the default one) and the value is a dictionary of '
    "keyword arguments for the template. Repeat the option for more groups; a group named twice keeps "
    "only the last occurrence, so pass all of a group's keys together."
)
# The pattern options accepting a file path all say so the same way.
PATH_FORM_HELP = " The pattern may also be given as a path to a YAML/JSON file holding it."


def object_pattern_help(
    subject: str, symbol: str, *, keyed: bool = False, call: bool = True, lead: str = "The object pattern"
) -> str:
    """Build the help text documenting the object pattern accepted for `subject`.

    Args:
        subject (str): What the pattern instantiates, written with its article so that it reads both as
            "used to instantiate <subject>" and as "if <subject> is defined as ...", e.g. "the model".
        symbol (str): The symbol used in the example, e.g. "MyModel".
        keyed (bool): Whether the pattern is keyed by a name, e.g. "model_name: [_obj_, ...]".
        call (bool): Whether the example object is called, adding the "_call_" entry to the pattern.
        lead (str): How the first sentence opens, for an option whose pattern is one of several accepted
            values, e.g. "Or the object pattern".

    Returns:
        str: The help text documenting both accepted spellings of the object pattern.
    """
    prefix = "model_name: " if keyed else ""
    call_part = ", {_call_: {...}}" if call else ""
    definition = "(...)" if call else ""
    return (
        f"{lead} used to instantiate {subject}. "
        f"For example, if {subject} is defined as `my_package.{symbol}{definition}`, then the pattern should be "
        f'"{prefix}[_obj_, {{_addr_: my_package.{symbol}, _file_: my_package.py}}{call_part}]" or '
        f'"{prefix}[_obj_, [_addr_, my_package.{symbol}, my_package.py]{call_part}]".'
    )


def template_param_option(example: str) -> Any:
    """Build the `--parameter` option, appending the caller's example sentence to the shared prose.

    Args:
        example (str): The command's own example sentence, without a leading space, e.g.
            'For example: --parameter "model: {input_size: 128}"'.

    Returns:
        Any: The typer `Option` declaration for `--parameter`.
    """
    return Option(None, "--parameter", "-p", parser=dict_parser, help=TEMPLATE_PARAM_HELP + " " + example)


def shapes_help(compact_example: str, init_example: str) -> str:
    """Build the shared `--shape` prose, without the sentence describing the INPUT_SHAPES fallback.

    Args:
        compact_example (str): The compact-form example including its quotes, e.g. '"image: [3, 224, 224]"'.
        init_example (str): The initializer shown in the explicit form, e.g. "torch.zeros".

    Returns:
        str: The shared help text; call sites append their own fallback sentence.
    """
    return (
        "Input tensor shapes for one sample, without the batch dimension, as a YAML mapping of input name to "
        f"specification. Compact form: {compact_example}. "
        f'Explicit form: "tokens: {{_SHAPE_: [512], _DTYPE_: int64, _INIT_: {init_example}}}". '
        "Specifications may nest in mappings and lists. _DTYPE_ defaults to bfloat16 (not float32), and an integer "
        "dtype without _INIT_ falls back to zeros with a warning."
    )


def shapes_option(help_text: str) -> Any:
    """Build a `--shape` option carrying `help_text`, typically composed from `shapes_help`.

    Args:
        help_text (str): The full help text, e.g. `shapes_help(...)` plus the command's fallback sentence.

    Returns:
        Any: The typer `Option` declaration for `--shape`.
    """
    return Option(None, "--shape", "-s", parser=tensor_shape_parser, help=help_text)


def compile_option(api: str) -> Any:
    """Build the `--compile` option for the frameworks that compile the model graph.

    Keras is not one of them: its `--compile` is run configuration, so it keeps its own declaration.

    Args:
        api (str): The compilation entry point, e.g. "torch.compile" or "nnx.jit".

    Returns:
        Any: The typer `Option` declaration for `--compile`.
    """
    return Option(
        None,
        "--compile",
        "-c",
        parser=bool_or_path_or_dict_parser,
        help=f'Whether to compile the model using "{api}". Omitted or false leaves the model uncompiled; '
        "true compiles with default options. Can also be a path to an existing YAML/JSON file, or a dictionary of "
        f'keyword arguments for "{api}".',
    )


def matmul_precision_option(detail: str = "") -> Any:
    """Build the `--matmul-precision` option, appending *detail* to the sentence on the faster modes.

    Args:
        detail (str): What the framework adds about the faster modes, without a trailing period, e.g.
            ", the latter by computing in bfloat16".

    Returns:
        Any: The typer `Option` declaration for `--matmul-precision`.
    """
    return Option(
        "high",
        envvar="MATMUL_PRECISION",
        help='Precision for float32 matrix multiplications: "highest" keeps full float32, while "high" and '
        f'"medium" trade accuracy for tensor-core speed{detail}.',
    )


def resume_option(restores: str, unsaved: str) -> Any:
    """Build the `--resume` option for the frameworks whose runs save a training state.

    Args:
        restores (str): What the restore puts back, plus whatever the framework adds about it, e.g.
            "Restores models, optimizers, grad scalers, and continues from the saved epoch.".
        unsaved (str): The run state the saved file does not carry, e.g. "data-order, sampler or RNG".

    Returns:
        Any: The typer `Option` declaration for `--resume`.
    """
    return Option(
        None,
        "--resume",
        help="Training state to resume from, in a form the active --logger understands: a local path always "
        "works, 'runs:/<run_id>/<artifact>' requires --logger mlflow, and "
        "'wandb://<entity>/<project>/<run_id>/<file>' requires --logger wandb; resuming across services is not "
        f"supported. {restores} Resume is exact only at epoch boundaries: the saved state carries the epoch, "
        f"step and update counters but no {unsaved} state, so the resumed epoch restarts from the beginning of "
        "the dataset.",
    )


def seed_option(detail: str = "") -> Any:
    """Build the `--seed` option, appending *detail* to the shared one-liner.

    Args:
        detail (str): What the framework adds about what the seed decides, with a leading space.

    Returns:
        Any: The typer `Option` declaration for `--seed`.
    """
    return Option(42, envvar="SEED", help=f"Random seed for reproducibility.{detail}")


def trainer_option(call: str, base: str) -> Any:
    """Build the `--trainer` option, whose contract differs only in the call and the intended base class.

    Args:
        call (str): The call the built trainer must accept, e.g. "trainer(learner=..., tracker=...)".
        base (str): The trainer class subclassing is the intended way to satisfy it, e.g. "FlaxTrainer".

    Returns:
        Any: The typer `Option` declaration for `--trainer`.
    """
    return Option(
        None,
        "--trainer",
        parser=path_or_any_parser,
        help=object_pattern_help("the trainer", "MyTrainer", call=False)
        + PATH_FORM_HELP
        + f" The pattern must build a callable accepted as {call} whose result exposes training_prefix, "
        "validation_prefix, callbacks, describe() and fit(epochs, start_epoch, validation_frequency); "
        f"subclassing {base} is the intended way.",
    )


def training_dataset_option(tail: str = "") -> Any:
    """Build the `--training-dataset` option, appending *tail* to the shared object-pattern prose.

    Args:
        tail (str): What the framework adds about how the batches reach the Learner, with a leading space.

    Returns:
        Any: The typer `Option` declaration for `--training-dataset`.
    """
    return Option(
        ...,
        "--training-dataset",
        parser=path_or_any_parser,
        help=object_pattern_help("the training dataset", "MyDataset") + PATH_FORM_HELP + tail,
    )


output_script_path = Option(
    None,
    "--output",
    "-o",
    help='Path of the generated Python script. Defaults to the snake_case --classname plus ".py" in the current '
    "directory. An existing file is overwritten and missing parent directories are created.",
)
model_pattern = Argument(
    parser=path_or_any_parser,
    help=object_pattern_help("the model", "MyModel") + PATH_FORM_HELP,
)
warmup_runs = Option(2, "--warmup-runs", "-w", help="Number of warmup runs before measuring inference time.")
times = Option(10, "--times", "-t", help="Number of iterations to measure the inference time.")
batch_size = Option(1, "--batch-size", "-b", help="Batch size for the input tensors during inference time measurement.")
# torch and keras; flax carries its own nnx.view variant in cmd_flax.py
training_mode = Option(
    False,
    help="Whether to set the model to training mode during inference time measurement. "
    "This can affect the inference time due to differences in behavior (e.g., dropout, batch norm).",
)


# The `train` options torch, flax and keras spell identically.
learner_pattern = Option(
    ...,
    "--learner",
    "-L",
    parser=path_or_any_parser,
    help=object_pattern_help("the learner class", "MyLearner")
    + PATH_FORM_HELP
    + " The factory is called with one keyword argument per positional model pattern, so its parameter names "
    "must match those model names.",
)
learner_outputs = Option(
    None,
    "--learner-outputs",
    "-LO",
    help="Criterion names the Learner's steps produce; they build the tracker and the progress-bar rows. "
    "Overrides the names the Learner declares itself, and is required when it declares none. Give the "
    'unprefixed names: the criterion options then take "loss" for the training criterion and "val_loss" for '
    "the validation one.",
)
epochs = Option(
    1,
    "--epochs",
    "-e",
    help="Epoch number to train up to, inclusive: training runs from --start-epoch (or the resumed epoch) "
    "through this number, so it is a count only when starting at epoch 1 and must not be smaller than the "
    "starting epoch. A resumed run must set it above the epoch stored in the resumed state.",
)
start_epoch = Option(
    1,
    help="First epoch number to run, 1-based and at most --epochs. It only offsets the loop counter; no data "
    "is skipped. Ignored when --resume is given, which continues at the epoch stored in the resumed state "
    "plus one.",
)
validation_dataset_pattern = Option(
    None,
    "--validation-dataset",
    "-V",
    parser=path_or_any_parser,
    help=object_pattern_help("the validation dataset", "MyDataset") + PATH_FORM_HELP,
)
validation_frequency = Option(
    1,
    "--validation-frequency",
    "-f",
    help="Run validation every N epochs; must be at least 1. On epochs where validation does not run, no "
    '"val_" criterion is produced, so val_-named best and save criteria are only monitored on validated epochs.',
)
lower_criteria = Option(
    ...,
    "--lower-criterion",
    "-LC",
    default_factory=list,
    show_default=False,
    help="Criterion names whose lower values are better, monitored for their lowest value. Name them as they "
    'appear in the epoch logs: training criteria keep the Learner\'s names, validation criteria carry the "val_" '
    'prefix (e.g. "val_loss"). A name no epoch produces is silently never monitored, and omitting the option '
    "monitors nothing.",
)
higher_criteria = Option(
    ...,
    "--higher-criterion",
    "-HC",
    default_factory=list,
    show_default=False,
    help="Criterion names whose higher values are better, monitored for their highest value. Name them as they "
    'appear in the epoch logs: training criteria keep the Learner\'s names, validation criteria carry the "val_" '
    'prefix (e.g. "val_accuracy"). A name no epoch produces is silently never monitored, and omitting the option '
    "monitors nothing.",
)
save_criteria = Option(
    ...,
    "--save-criterion",
    "-SC",
    default_factory=list,
    show_default=False,
    help='Criterion names whose best-scoring model states are saved, as a "best_<criterion>" artifact. Each '
    "name must also be given to --lower-criterion or --higher-criterion, spelled the same way; a name in "
    "neither list is silently ignored, and omitting the option saves no best-model artifact.",
)
experiment = Option("experiment", "--experiment", "-E", envvar="EXPERIMENT", help="Experiment name for the logger.")
logger_name = Option("mlflow", "--logger", help="Experiment tracking service to record the run to.")
log_arguments = Option(
    None,
    "--log-arguments",
    "-K",
    parser=dict_parser,
    help='Extra key-value pairs to record in the run\'s "arguments.yaml" artifact, in the format "key: value". '
    "Repeat the option to add more keys; a key given twice keeps only the last occurrence, and the keys the run "
    "records itself take precedence over keys of the same name.",
)
log_artifacts = Option(
    None,
    "--log-artifacts",
    "-A",
    help="Paths to files to upload as run artifacts. Repeat the option to add more.",
)
ci = Option(
    False,
    help="Whether to run in CI mode. "
    "If true, it will print the criteria at the end of each epoch instead of using a progress bar.",
)

__all__ = [
    "PATH_FORM_HELP",
    "batch_size",
    "ci",
    "compile_option",
    "epochs",
    "experiment",
    "higher_criteria",
    "learner_outputs",
    "learner_pattern",
    "log_arguments",
    "log_artifacts",
    "logger_name",
    "lower_criteria",
    "matmul_precision_option",
    "model_pattern",
    "object_pattern_help",
    "output_script_path",
    "resume_option",
    "save_criteria",
    "seed_option",
    "shapes_help",
    "shapes_option",
    "start_epoch",
    "template_param_option",
    "times",
    "trainer_option",
    "training_dataset_option",
    "training_mode",
    "validation_dataset_pattern",
    "validation_frequency",
    "warmup_runs",
]

if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
