# Checkpoints go through the DCP state-dict API; resume is an epoch-boundary loader

This ADR records how training state is produced, written, and loaded back, replacing the hand-rolled
`_get_state_dict` / `_unwrap_ddp` pair and adding the resume path that never existed (nothing in the
package ever read `training_state` back; `--start-epoch` only offset the loop counter).

## One producer: `torch.distributed.checkpoint.state_dict`

`get_model_state_dict` / `get_optimizer_state_dict` with `StateDictOptions(full_state_dict=True,
cpu_offload=True)` return wrapper-free keys for raw, compiled, DDP-wrapped, and `fully_shard`'d models
alike — subsuming `_unwrap_ddp`, which was a no-op on the dispatch path (it received raw modules) and
insufficient on the interrupt path (it cannot strip `_orig_mod.`). Optimizer state is keyed by parameter
FQNs, which requires the model↔optimizer pairing that generated learners now expose as
`optimizer_models`; without the pairing, plain optimizer `state_dict` keys are used with a warning, except
under FSDP2 where sharded state is unresolvable without FQNs and the strategy fails loud instead. Old torch
without the DCP module falls back to plain state dicts with wrapper prefixes stripped.

## Produce on every rank, write on rank 0

Gathering a full state dict is a collective: a rank-0-only producer hangs the job under FSDP2. The saver
callbacks (`TrainingStateSaver`, best-criterion saving) are therefore constructed on every rank and call
`strategy.state_dict` everywhere; only rank 0 holds a logger and writes the artifact. This relies on every
rank reaching the save decision with the same answer — true because best-criterion comparisons run on
tracker values that are all-reduced — an invariant now stated rather than assumed.

## The interrupt-time save is deleted

Saving from a `KeyboardInterrupt` handler was already broken (it saved the CLI's wrapped copies, producing
`_orig_mod.*` keys no loader accepts) and cannot be made collective-safe: a signal lands on an arbitrary
rank at an arbitrary point, so a collective save from the handler is a deadlock window. The ecosystem
agrees — timm passes on KeyboardInterrupt and offers periodic recovery saves; NeMo and Lightning demote the
signal to a flag, broadcast it, and stop at a synchronized step boundary before saving. If mid-epoch loss
protection is wanted later, the cheap correct shape is a periodic step-interval saver riding the same
collective-safe producer, not an interrupt handler.

## Resume restores state at epoch boundaries

`--resume` accepts a local path, an MLflow `runs:/` URI, or a `wandb://` reference; rank 0 fetches and
`torch.load`s the artifact, then `strategy.load_state_dict` distributes it: model tensors travel through
`set_model_state_dict(broadcast_from_rank0=True)`, while optimizer states and metadata are object-broadcast
to every rank first — the broadcast option cannot infer a device from stateless optimizers (plain SGD), for
which only hyperparameters are restored. Gradient-scaler states and the loop counters come from the same
artifact; the run continues at `meta.epoch + 1`, overriding `--start-epoch` with a warning. Initializers and
the initial-weight broadcast are skipped on resume since the loaded state overwrites them. Sampler and
dataloader positions are not saved, so resume is exact only at epoch boundaries; step-exact resume would
need RNG and sampler state in `meta` and is out of scope.
