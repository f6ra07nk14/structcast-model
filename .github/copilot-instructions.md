# StructCast-Model Development Guide

## Copilot Interaction Policy

- **ALWAYS respond in English**, regardless of the language used in the user's request
- Non-English requests should be understood and acknowledged, but all responses, explanations, plans, and code comments must be in English

## Architecture

Configuration-driven toolkit that generates PyTorch models and training workflows from YAML templates, built on [StructCast](https://github.com/f6ra07nk14/structcast). See [README_AGENT.md](../README_AGENT.md) for the full data-flow diagram and CLI surface.

Key modules:
- `builders/schema.py` — Pydantic schemas for YAML templates
- `builders/base_builder.py` → `builders/torch_builder.py` — Template-to-code pipeline
- `commands/main.py` + `commands/cmd_torch.py` — Typer CLI (`scm` entry point)
- `base_trainer.py` — Generic trainer, callbacks, best-criterion tracking
- `torch/trainer.py` — PyTorch training steps, tracker, EMA, timm wrappers, distributed training via `torchrun` + DDP

## Build & Test

```bash
# Install (uv required)
uv sync --extra torch-cpu --dev --group tox

# Run all checks (type check + lint + test)
tox -e py311

# Individual steps
pytest                          # tests with coverage
mypy src && mypy tests          # type checking
ruff format --check src tests   # format check
ruff check src tests            # lint
ruff format src tests           # auto-format
```

## Code Style

- **Python ≥ 3.11**, line length 120 (`ruff`)
- **Google-style docstrings** — enforced by `ruff` pydocstyle
- **Explicit `__all__`** in every module
- **Absolute imports** — `from structcast_model.xxx import yyy`
- **Lazy imports** in `__init__.py` via `LazySelectedImporter` (guarded by `TYPE_CHECKING`)
- **Type annotations on all functions** — enforced by `mypy` (`disallow_untyped_defs = true`)
- Use `collections.abc` types (`Mapping`, `Iterable`, `Callable`) and union syntax (`X | Y`)
- Logger per module: `logger = getLogger(__name__)`

## Testing Conventions

- Files: `test_<module>.py` — Functions: `test_<feature>_<scenario>`
- Every test function has a return type annotation `-> None` and a one-line Google docstring
- Use `pytest.raises(ExcType, match=...)` for exception testing
- Use `@pytest.mark.parametrize` for data-driven tests
- Shared fixtures live in `tests/conftest.py`
- Tests run with `--doctest-modules` — keep doctests in source valid

## Conventions

- Raise specific exceptions (`ValueError`, `KeyError`) with descriptive f-string messages
- Validate inputs at function entry; log warnings for recoverable issues
- YAML templates in `cfg/` follow StructCast object patterns (`_obj_`, `_addr_`, `_file_`, `_call_`, `_bind_`, `_attr_`)
- Generated code is re-imported at runtime via `_file_` patterns — do not break this loop
- `scm torch train` supports distributed training via `torchrun` + `DistributedDataParallel` — use `torchrun --nproc_per_node=gpu -m structcast_model.commands.main torch train ...` for multi-GPU
