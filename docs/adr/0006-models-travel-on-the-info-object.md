# Models travel on the info object, not as event kwargs

Every lifecycle event used to pass the learner's models as keyword arguments — `on_*(info, **models)` and
`on_best(info, best, **models)` — forcing all 42 event implementations in the repository to declare `**models`
while only three production callbacks consumed them (`BestCriterion.on_epoch_end` forwarding to `on_best`, the
torch best-model logger, and the torch training-state saver). Models now live on the info object: `BaseInfo` is
generic and exposes a `models` property (empty on a bare info), `BaseTrainer` overrides it to delegate to
`self.learner.models`, and every event signature shrinks to `on_*(info)` / `on_best(info, best)`.

## One invariant type variable threads the whole chain

Moving models onto `info` reverses the direction the type variable flows: `ModelT_contra` was legal only because
models appeared exclusively in parameter position, and a contravariant type variable cannot appear in a return
position, so `BaseInfo(Generic[ModelT_contra])` is ill-typed on arrival; `dict[str, T]` being invariant rules out
the covariant escape as well. The chain therefore uses a single invariant `ModelT`:
`Learner[ModelT].models -> dict[str, ModelT]`, `BaseInfo[ModelT].models -> dict[str, ModelT]`,
`BaseTrainer(BaseInfo[ModelT])` with `learner: Learner[ModelT]`, and every event protocol taking
`info: BaseInfo[ModelT]`. `on_best` tightens its second parameter from `BestCriterion[Any]` to
`BestCriterion[ModelT]` — the `Any` was historical, and the torch implementation already annotated the concrete
type.

A dual-type-variable scheme — covariant providers returning `Mapping[str, ModelT_co]` plus contravariant consumer
protocols — was rejected: it keeps cross-specialization callback reuse type-checking (an `OnUpdate[nn.Module]`
callback against a hypothetical `BaseTrainer[ResNet]`), but costs three type variables coexisting in one file and
a `Mapping` return type, for flexibility no consumer uses — the only production instantiation is
`BaseTrainer[torch.nn.Module]`. Two accepted consequences: the event protocols lose contravariance, and the
`dict` return leaves `info.models` writable, so a callback mutating it would corrupt the learner's own dict —
policed by convention, not by types.

## Live delegation instead of a snapshot

Dispatch used to read `learner.models` once per `train()` / `evaluate()` / `fit()` and pass that same dict to
every event of the phase. `info.models` now reads `learner.models` on every access. The snapshot was an
implementation accident, never a promise; live delegation removes a piece of state the trainer would otherwise
have to keep synchronized, at the cost that a callback observes whatever the learner currently holds.

This is a breaking release. At runtime an out-of-tree callback still declaring `**models` keeps being called
(the kwargs arrive empty), but one that actually read a model kwarg breaks — a subscript raises `KeyError`, a
`.get()` reader silently sees nothing — the migration is reading `info.models`. A deprecation period (dispatching kwargs one more major version behind a warning) was
rejected: detecting which callbacks consume models outweighs the refactor itself, and the in-repo consumers all
migrate in the same commit.
