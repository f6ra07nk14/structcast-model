"""Utilities shared by builder modules."""

import ast
from collections import defaultdict
from hashlib import sha256
from json import dumps as json_dumps
from re import compile as re_compile
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from pydantic_core import to_jsonable_python
from structcast.core.constants import SPEC_SOURCE
from structcast.core.exceptions import SpecError
from structcast.core.instantiator import AddressPattern, AttributePattern, BindPattern, CallPattern, ObjectPattern
from structcast.core.specifier import SPEC_CONSTANT, SpecIntermediate
from structcast.utils.base import resolve_address

from structcast_model.builders.constants import BOUND_CALLABLE_PREFIX, FILE_IMPORT_PREFIX
from structcast_model.builders.schema import SPEC_EVAL

_MODULE_LEVEL_NAME = re_compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
"""A resolved expression that is nothing but a name: everything `_resolve` names lives at module level."""


def _hoist(imports: defaultdict[str, set[str | None]], expression: str, leaf: str) -> str:
    """Collect one closure-free expression as a module-level constant and return the name it takes.

    The name carries the digest of the expression, so it is the same in every script the same
    binding is resolved for -- deterministic across processes, and one constant however many layers,
    classes or learners bind that callable to those arguments.
    """
    name = f"_bound_{leaf}_{sha256(expression.encode()).hexdigest()[:8]}"
    imports[f"{BOUND_CALLABLE_PREFIX}{name}"].add(expression)
    return name


def _literal_arguments(rendered: str) -> bool:
    """Whether a rendered argument list carries build-time literals and nothing else.

    Read off the rendered arguments rather than the pattern, because that is where an `eval:` value,
    a nested pattern or a source reference has already turned into a name or a call: anything that
    survives `ast.literal_eval` was a constant in the configuration file.
    """
    try:
        call = ast.parse(f"_({rendered})", mode="eval").body
        if not isinstance(call, ast.Call):
            return False
        for node in [*call.args, *(keyword.value for keyword in call.keywords)]:
            ast.literal_eval(node)
    except (SyntaxError, ValueError):
        return False
    return True


def statement_names(line: str) -> tuple[set[str], set[str]]:
    """Return the (loaded, stored) variable names of one generated statement."""
    loads: set[str] = set()
    stores: set[str] = set()
    for node in ast.walk(ast.parse(line.strip())):
        if isinstance(node, ast.Name):
            (stores if isinstance(node.ctx, ast.Store) else loads).add(node.id)
    return loads, stores


def stored_names(output: str) -> list[str]:
    """Return the variable names one flow step assigns, unpacking the `(a, b)` form of a multi-output step."""
    return [name.strip() for name in output.strip("()").split(",") if name.strip()]


def resolve_object(imports: defaultdict[str, set[str | None]], pattern: ObjectPattern) -> tuple[str, str]:
    """Resolve the object pattern to a string representation and collect the required imports.

    The object pattern is resolved by processing its patterns in order.
    The first pattern must be an `AddressPattern` or an `ObjectPattern`, which is resolved to a string representation.
    The subsequent patterns are applied to the resolved string representation in order,
    where an `AttributePattern` is resolved to an attribute access, a `CallPattern` is resolved to a function call,
    and a `BindPattern` is resolved to a function call with the bound arguments.
    The required imports are collected from the `AddressPattern`s encountered during the resolution process.

    Args:
        imports (defaultdict[str, set[str | None]]): A dictionary to collect the required imports,
            where the keys are module names and the values are sets of imported names from the corresponding modules.
        pattern (ObjectPattern): The pattern to resolve.


    Returns:
        tuple[str, str]: A tuple containing the resolved string representation of the object pattern and
            the name of the top-level class or function.
    """
    classes: list[str] = []
    # Every `eval:` value rendered so far, so a binding can tell whether it read one of them.
    evaluated: list[str] = []

    def _repr(raw: Any) -> str:
        if isinstance(raw, (int, float, bool, bytes, type(None))):
            return repr(raw)
        try:
            return _resolve(ObjectPattern.model_validate(raw))
        except ValidationError:
            pass
        if isinstance(raw, str):
            if raw.startswith("eval:"):
                evaluated.append(raw)
                return raw[5:].strip()
            return repr(raw)
        if isinstance(raw, dict):
            return f"{{{', '.join(f'{_repr(k)}: {_repr(v)}' for k, v in raw.items())}}}"
        if isinstance(raw, (list, tuple)):
            return f"[{', '.join(_repr(item) for item in raw)}]"
        raise SpecError(f"Unsupported type for validation: {type(raw)}")

    def _args(raw: Any) -> str:
        if isinstance(raw, dict):
            return ", ".join(f"{k}={_repr(v)}" for k, v in raw.items())
        if isinstance(raw, (list, tuple)):
            return ", ".join(_repr(v) for v in raw)
        return _repr(raw)

    def _resolve(obj: ObjectPattern) -> str:
        first, rest = obj.patterns[0], obj.patterns[1:]
        if isinstance(first, AddressPattern):
            module, res = resolve_address(first.address)
            classes.append(res)
            if first.file:
                # File-addressed objects cannot be imported by module name: record the file under a
                # special key so the script renderer emits an import_from_address binding instead.
                imports[f"{FILE_IMPORT_PREFIX}{first.file}"].add(first.address)
            elif module:
                imports[module].add(res)
        elif isinstance(first, ObjectPattern):
            res = _resolve(first)
        else:
            raise SpecError(
                "First pattern of an ObjectPattern must be an AddressPattern or ObjectPattern "
                f"but got: {to_jsonable_python(pattern)}"
            )
        for bind_index, ptn in enumerate(rest):
            if isinstance(ptn, (AddressPattern, ObjectPattern)):
                raise SpecError(
                    "Only the first pattern of an ObjectPattern can be an AddressPattern or ObjectPattern "
                    f"but got: {to_jsonable_python(pattern)}"
                )
            if isinstance(ptn, AttributePattern):
                res = f"{res}.{ptn.attribute}"
            elif isinstance(ptn, CallPattern):
                res = f"{res}({_args(ptn.call)})"
            elif isinstance(ptn, BindPattern):
                # The position in `rest` is deterministic, unlike an id()-derived suffix, so the same
                # pattern always renders the same script. Reuse across nesting levels is safe: a
                # nested lambda only ever references its own arguments, shadowing any outer ones.
                aname, kwname = f"_arg{bind_index}", f"_kw{bind_index}"
                seen, target = len(evaluated), res
                args = _args(ptn.bind)
                if isinstance(ptn.bind, dict):
                    res = f"(lambda *{aname}, **{kwname}: {target}(*{aname}, {args}, **{kwname}))"
                else:
                    res = f"(lambda *{aname}, **{kwname}: {target}({args}, *{aname}, **{kwname}))"
                # A closure over nothing but constants and one module-level name is hoisted, so that
                # every layer binding that callable to those arguments shares the one object: built
                # per instance instead, it would be a per-instance leaf of the Flax graphdef, and two
                # instances of the generated class would no longer hit the same `flax.nnx.jit` trace.
                # An `eval:` value is written to be read where the object is built -- `rngs` is the
                # standing example -- so a binding that read one stays where the reading works.
                if len(evaluated) == seen and _MODULE_LEVEL_NAME.fullmatch(target) and _literal_arguments(args):
                    res = _hoist(imports, res, target.rsplit(".", 1)[-1])
            else:
                raise SpecError(
                    "Patterns after the first pattern of an ObjectPattern must be AttributePattern, CallPattern, "
                    f"or BindPattern but got: {to_jsonable_python(pattern)}"
                )
        return res

    return _resolve(pattern), (classes[0] if classes else "_Class")


def optimizer_hash(optimizer: ObjectPattern) -> str:
    """Return the digest identifying one `OPTIMIZER` pattern, as it was written.

    Recorded in the generated learner and in the training state so a resume can report an optimizer
    that was rebuilt from a different configuration (`docs/adr/0015`): the learner builds the
    optimizer from the pattern and the restored state cannot see it, so a swapped schedule would
    otherwise continue silently from the old step count. Framework-neutral -- it hashes the
    validated pattern and nothing else -- so the Flax and Keras builders emit comparable digests;
    on the Flax side the pattern is hashed before `inject_learning_rate` rewrites it, so turning the
    injection on or off never moves the digest.

    Args:
        optimizer (ObjectPattern): The validated `OPTIMIZER` pattern of one learner behavior.

    Returns:
        str: The hex SHA-256 of the pattern's canonical JSON dump.
    """
    dumped = optimizer.model_dump(by_alias=True)
    return sha256(json_dumps(dumped, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def resolve_getter(imports: defaultdict[str, set[str | None]], spec: Any, variable: str | None = None) -> str:
    """Resolve the given specification to a string representation and collect the required imports if applicable.

    Args:
        imports (defaultdict[str, set[str | None]]): A dictionary to collect the required imports,
            where the keys are module names and the values are sets of imported names from the corresponding modules.
        spec (Any): The specification to resolve.
        variable (str | None): The variable name to use for source identifier specifications
            if the source identifier does not have a single string index as its value.
            If not provided, the variable name will be resolved from the value of the source identifier.

    Returns:
        str: The resolved string representation of the specification.
    """

    def _getter(raw: Any, var_name: str | None = None) -> str:
        try:
            return resolve_object(imports, ObjectPattern.model_validate(raw))[0]
        except ValidationError:
            pass
        if isinstance(raw, dict):
            return f"{{{', '.join(f'{repr(k)}: {_getter(s, var_name)}' for k, s in raw.items())}}}"
        if isinstance(raw, list):
            return f"[{', '.join(_getter(s, var_name) for s in raw)}]"
        if isinstance(raw, tuple):
            return f"({', '.join(_getter(s, var_name) for s in raw)})"
        if not isinstance(raw, str):
            return repr(raw)
        spec = SpecIntermediate.convert_spec(raw)
        if spec.identifier == SPEC_SOURCE:
            var_name, attr = (var_name, spec.value) if var_name else (spec.value[0], spec.value[1:])
            return f"{var_name}{''.join(f'[{repr(s)}]' for s in attr)}"
        if spec.identifier in SPEC_EVAL:
            return spec.value
        if spec.identifier == SPEC_CONSTANT:
            return repr(spec.value)
        raise SpecError(f"Unsupported spec identifier: {spec.identifier}")

    return _getter(spec, variable)


__all__ = ["optimizer_hash", "resolve_getter", "resolve_object", "statement_names", "stored_names"]

if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
