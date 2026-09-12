# Loggers live in a framework-agnostic package and import frameworks lazily

The `Logger` protocol (`log_params`, `log_metric`, …, `log_state_dict`, `fetch_training_state`) describes an
experiment-tracking run, not a PyTorch concept, and the planned flax and keras trainers need the same interface.
Today it is buried in `structcast_model/torch/` and hard-imports `torch` at module level. We move the three logger
modules into their own package and make every framework import lazy:
`torch/{logger,mlflow_logger,wandb_logger}.py` becomes `loggers/{base,mlflow,wandb}.py`. The same naming rule
then drops the redundant suffixes from the builder modules (`builders/torch_builder.py` → `builders/torch.py`).

## Framework imports are lazy; only the service SDKs stay behind `try_import`

`torch` in all three modules and `mlflow.pytorch` in `loggers/mlflow.py` are bound through
`structcast.utils.lazy_import.LazyModuleImporter`, with the real imports kept under `TYPE_CHECKING` for type
checkers. `torch` has only four call sites, all `torch.load`/`torch.save`. `mlflow.pytorch` needs the same
treatment for a subtler reason: its top level pulls in `torch`, so importing it eagerly would defeat the laziness
of every logger whenever mlflow is installed (verified against mlflow 3.15.1: `mlflow/pytorch/__init__.py` imports
`mlflow/pytorch/_lightning_autolog.py`, whose top level imports `torch`). `try_import()` keeps guarding only the
service SDKs (`mlflow`, `wandb`), whose absence is the case the constructor should explain.

Accepted trade-off: with mlflow installed but torch missing, the error moves from construction time
(`_imports.check()`) to the first `log_state_dict` / `fetch_training_state` call. A laziness regression test —
a subprocess asserting `"torch" not in sys.modules` after importing the three `loggers` modules — guards this, since
the property is invisible in ordinary use and a single stray top-level import silently reverts it. Future flax and
keras logger backends use the same `LazyModuleImporter` pattern.

## Package `__init__` files are lazy submodule routers

`loggers/__init__.py` and `torch/__init__.py` adopt the `LazySelectedImporter` + `import_structure` pattern already
used by the top-level `structcast_model/__init__.py`. Every module file lists its public symbols, so they are also
reachable flat on the package (`structcast_model.torch.TorchTrainer`, `structcast_model.loggers.Logger`); subpackage
entries stay submodule-only with an empty list. A name re-exported by several modules is listed exactly once — under
the module that defines it when that module has an entry of its own, otherwise under the module that re-exports it —
because `_class_to_module` is a dict comprehension and keeps the last writer. Hence `initial_distributed_env` routes to
`torch/distributed.py`, not to `torch/trainer.py`, while `CriteriaTracker`, defined inside the submodule-only
`torch/layers/` subpackage, routes through the `torch/trainer.py` that re-exports it. Consumers then
hold one lazy binding per package instead of one per module — `commands/cmd_torch.py` drops five
`LazyModuleImporter` bindings for `scm_loggers` and `scm_torch`, reached by flat access
(`scm_loggers.Logger`, `scm_torch.TorchTrainer`). The two handles are plain imports of
the package shims — wrapping a shim in `LazyModuleImporter` does not work, because its first access
copies the shim's still-unresolved submodule slots (`None`) into the wrapper's `__dict__`, so every
later access returns `None`.

ADR-0004's exemption is untouched: `structcast_model.torch.distributed` itself stays un-shimmed, because generated
step code imports the sync gate from it inside dynamo's tracing path. Only the package `__init__` becomes a shim,
and the package object is not on that path.

## `builders` modules drop their suffixes

For naming uniformity with the new package, `builders/{base_builder,flax_builder,keras_builder,torch_builder}.py`
become `builders/{base,flax,keras,torch}.py`. Naming a submodule after a third-party top-level package
(`builders/torch.py`, `loggers/mlflow.py`, `loggers/wandb.py`) is an established ecosystem convention —
`safetensors.torch`, `einops.layers.torch`, `wandb.keras`, `mlflow.pytorch` — and Python 3 absolute imports make it
safe. The guard is that `builders/__init__.py` must never define a same-named attribute, and that
`README_AGENT.md` teaches the `from structcast_model.builders import torch as torch_builder` import style.
ADR-0003's reference to `torch_builder.py` carries a rename note pointing here; its reasoning is unaffected.

## Rejected: compatibility aliases, loggers under `torch/`, an eager `mlflow.pytorch`

- **Back-compatibility aliases or a deprecation period for the old module paths.** None are added; this is one
  breaking release, consistent with the recorded policy of ADR-0002 (the `Learner` rename) and ADR-0006 (models on
  the info object). Two live spellings of every logger import is a worse steady state than one migration.
- **Keeping the loggers under `torch/`.** The protocol is framework-agnostic and its only torch dependency is
  checkpoint serialization; leaving it there would force the flax and keras trainers to import the torch package
  for an interface that has nothing to do with torch.
- **Importing `mlflow.pytorch` eagerly inside `try_import`.** Simpler to read, but wherever mlflow is installed it
  drags `torch` into every process that merely imports `loggers/mlflow.py` — exactly the cost the lazy binding
  exists to remove.
