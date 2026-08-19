# Training state flows through logger state backends as single-file archives

`log_state_dict` / `fetch_training_state` were the only `Logger` members with a framework baked in: three
`torch.load` sites, one `torch.save`, one `mlflow.pytorch.log_state_dict`, plus the `.pt`/`*.pth` filename
conventions (ADR-0011 counted them). Flax state is a different serialization problem — orbax, whose checkpoint
is always a directory — and Keras (#21) will be a third. The loggers gain one seam instead of per-framework
subclasses: a `StateBackend` with `save(states, directory, name) -> Path`, `load(path) -> dict`, and a filename
suffix, injected at logger construction and defaulting to the torch backend (today's five call sites, moved
verbatim).

The invariant both backends preserve is the existing torch contract: **fetch returns host-memory state; the
strategy owns device placement** (`map_location="cpu"` today). The Flax backend therefore:

- saves with orbax (`CheckpointManager` v0 API + `ocp.args.Composite`: model states, optimizer states, and a
  JSON `meta` item) into a temporary directory, waits for the async commit, then packs **one** `.tar.gz` — so
  the wandb single-file run-dir flow, the `wandb://…/<file>` and `runs:/…` reference formats, and
  `TrainingStateSaver`/`_BestLogger` all survive unchanged;
- loads by extracting with `tarfile.extractall(filter="data")` and restoring to host numpy (no sharding
  target), so a checkpoint saved on four devices restores on any topology; typed RNG keys travel as orbax's
  native `key_data` representation.

The trust boundary keeps its guard: torch mitigates pickle RCE with `weights_only=True`; orbax is pickle-free,
and the archive extraction uses the stdlib path-traversal filter. The payload contract is the torch one —
`{"models", "optimizers", "grad_scalers", "meta": {epoch, step, update}}` with `grad_scalers` always empty on
the Flax side — extended by Flax-only `meta` entries: the run seed, a hash of the resolved configuration, and a
per-segment hash of the resolved optimizer pattern.

## Resume policy

Aligned with the torch loader (ADR-0005): epoch-boundary only, `meta.epoch + 1`, initializer application
skipped on resume, and a **new** tracking run per invocation with the `--resume` reference logged for
provenance (run re-attachment is a separate cross-framework feature, not started here). Because optax rebuilds
`tx` from configuration and the state restore cannot see it, swapping only the LR schedule between save and
resume restores cleanly and silently continues the new schedule from the old count — an experimentally verified
footgun. The loader compares the saved per-segment optimizer hashes against the rebuilt configuration and
**warns** on mismatch (naming the segment) rather than refusing: extending a schedule or lowering the LR of a
fine-tune run is legitimate; silence is not.

## Trade-offs

- Archiving trades orbax's streaming atomicity and partial/lazy restore for a zero-change transport contract;
  at single-host ConvNeXtV2 scale the extra temporary copy is noise. If checkpoints grow to where this matters,
  the backend's internals can switch to directory artifacts without touching the seam.
- `mlflow.pytorch.log_state_dict` is replaced by a plain artifact upload for both backends, so mlflow's
  torch-flavored state layout (`state_dict.pth` inside a directory artifact) changes to the backend's single
  file; the fetch path accepts both via the suffix glob.
