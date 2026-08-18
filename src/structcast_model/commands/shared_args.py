"""Typer declarations shared by the torch, keras and flax sub-apps.

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

from typing import Any

from typer import Argument, Option

from structcast_model.commands.utils import bool_or_path_or_dict_parser, dict_parser, path_or_any_parser

TEMPLATE_PARAM_HELP = (
    "Parameters to format the template configuration file with. "
    'Each parameter is "group: {...}", where the group name selects the parameter group '
    '("SHARED" applies to every group, "DEFAULT" to the default one) and the value is a dictionary of '
    "keyword arguments for the template. Repeat the option for more groups; a group named twice keeps "
    "only the last occurrence, so pass all of a group's keys together."
)
# The pattern options accepting a file path all say so the same way.
PATH_FORM_HELP = " The pattern may also be given as a path to a YAML/JSON file holding it."


def object_pattern_help(subject: str, symbol: str, *, keyed: bool = False, call: bool = True) -> str:
    """Build the help text documenting the object pattern accepted for `subject`.

    Args:
        subject (str): What the pattern instantiates, written with its article so that it reads both as
            "used to instantiate <subject>" and as "if <subject> is defined as ...", e.g. "the model".
        symbol (str): The symbol used in the example, e.g. "MyModel".
        keyed (bool): Whether the pattern is keyed by a name, e.g. "model_name: [_obj_, ...]".
        call (bool): Whether the example object is called, adding the "_call_" entry to the pattern.

    Returns:
        str: The help text documenting both accepted spellings of the object pattern.
    """
    prefix = "model_name: " if keyed else ""
    call_part = ", {_call_: {...}}" if call else ""
    definition = "(...)" if call else ""
    return (
        f"The object pattern used to instantiate {subject}. "
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


__all__ = [
    "PATH_FORM_HELP",
    "TEMPLATE_PARAM_HELP",
    "batch_size",
    "compile_option",
    "model_pattern",
    "object_pattern_help",
    "output_script_path",
    "shapes_help",
    "template_param_option",
    "times",
    "training_mode",
    "warmup_runs",
]
