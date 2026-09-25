# Source comment ledger

The Python sources under `src/` carry `"""..."""` docstrings as their only prose. The notes that used to live
in `#` comments are kept here instead, one row per note: the file, the line of the code the note explains, that
code, and the note itself.

- Line numbers refer to the code as of the commit that introduced this ledger; later edits move them. The *Code*
  column is the anchor to search for when they drift.
- A note that sat on its own line(s) is anchored to the first code line below it; a trailing note is anchored to
  its own line.
- Tool directives (`# type: ignore[...]`, `# noqa: ...`) remain inline in the source because mypy and ruff only
  honour them there. Their justification, where the source gave one, is recorded here.
- A note that belongs in a docstring or an ADR should move there rather than grow this file.

## `src/structcast_model/base_trainer.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 11 | `from typing_extensions import Protocol, runtime_checkable` | Protocol and runtime_checkable come from typing_extensions so that isinstance checks use inspect.getattr_static on Python 3.11 as well (backported from 3.12): probing a protocol member must not execute a property getter, which for a data provider may build a real data loader. |
| 440 | `for callback in self.callbacks:` | The learner/tracker/data participants legitimately may implement no event, but an entry of the explicit callbacks sequence that matches nothing is almost certainly a typo'd hook name. |

## `src/structcast_model/builders/base.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 110 | `rendered_file = str(Path(file).resolve()) if Path(file).exists() else file` | Resolve the config-relative path while it is resolvable, so the generated script imports the same file regardless of the directory it is later run from. |
| 115 | `hoisted = "\n".join(` | Module level, so every instance of every generated class shares one bound callable object. |
| 255 | `user_defined_layer_type: ClassVar[type[LayerIntermediateT]] = cast(type[LayerIntermedia...` | Subclasses bind the type to the concrete intermediate they parametrize the builder with, which a `ClassVar` cannot express, so the default is cast to the type variable. Dropping `ClassVar` instead would turn the attribute into a per-instance dataclass field. |
| 370 | `merged = cast(Parameters, parameters.merge(unit.PARAM))` | `merge` is annotated to return the base `Parameters` while it instantiates `type(self)` at runtime. |
| 457 | `gradient_checkpointing=self._resolve_gradient_checkpointing(imports, module.GRADIENT_CH...` | Resolved after the flows: `collected_imports` keeps insertion order, so an earlier resolution would reorder the emitted import header of every checkpointed layer. |
| 604 | `user_defined_learner_layer_type: ClassVar[type[LearnerIntermediateT]] = cast(` | Subclasses bind the type to the concrete intermediate they parametrize the builder with, which a `ClassVar` cannot express, so the default is cast to the type variable. Dropping `ClassVar` instead would turn the attribute into a per-instance dataclass field. |
| 608 | `template_type: ClassVar[type[Template[Any]]] = TemplateLearner` | The template class decides which keys count as learner fields and which fall through to the layer builder, so a framework extending the learner schema must bind its own template here. |
| 756 | `segment.backward_kwargs = ", ".join(f"{k}={resolve_getter(imports, v)}" for k, v in lea...` | Rendered after the flow: `collected_imports` keeps insertion order, so resolving EXTRA any earlier reorders the emitted import header for every config that uses it. |

## `src/structcast_model/builders/flax.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 24 | `optimizer_hash,` | re-exported here, next to the builder that emits a learner's `OPTIMIZER_HASHES` |
| 88 | `base, forward = "structcast_model.flax.layers.GradientCheckpointingModule", "_forward"` | The base owns `__call__` and rematerializes the body it finds under `_forward`. |
| 174 | `imports["jax"].add(None)` | A bare name is one of the policies JAX ships; anything else -- a pattern building a parameterized policy, say -- resolves like any other DSL value. |
| 207 | `if node and any(_is_inject(node[0], value) for value in node[1:]):` | Addresses serialize either as `{"_addr_": ...}` or as the `["_addr_", ...]` list form. |
| 222 | `dumped = cast(list[object], ObjectPattern.model_validate(node).model_dump(by_alias=True))` | `ObjectPattern` serializes to the `["_obj_", <part>, ...]` list its validator accepts back, which the `model_dump` signature cannot express. |
| 235 | `static_args = [key for key in _keywords(parts[index]) or {} if key != "learning_rate"]` | `static_args` is the safety valve: without it inject arrayifies every numeric keyword, and `bool` is an `int` subclass, so a flag like `nesterov=True` would reach the factory as `Array(1)`. |
| 501 | `needed = {segment.loss, *self.outputs, *[name for reads in updates[index:] for name in ...` | Only what the enclosing step reads leaves the flow: the criteria, what any update expression from here on reads -- the `EXTRA` keywords are evaluated in the step, so a later one reads this flow's values there -- and what a later segment takes as a parameter. Every other intermediate stays local, so a flow may compute values a traced auxiliary output could not carry. |
| 510 | `scaled = f"{segment.loss} * _loss_scale" if segment.scale else segment.loss` | A scaled segment differentiates its loss multiplied by the scale it is handed, and reports the plain one as its criterion: what the scale keeps out of the float16 underflow range is the backward pass, not the number the run is judged by. |
| 514 | `argnums = f", argnums={tuple(range(len(owned)))}" if len(owned) > 1 else ""` | The owned models are the leading parameters, so their positions are the `argnums`; the default of 0 already names the single owned model of a one-module segment. |
| 527 | `if index == 0:` | The helper divides the scale back out, keeps the state it just wrote wherever a gradient came back non-finite, and reports the apply it attempted either way. |
| 671 | `body.append(f"self._ema_state_{name} = {self.others[f'ema_{name}']}")` | The view shares its variables with the average, so running it is running the average. |
| 818 | `base = BaseLearnerBuilder._build_segment(self, imports, module, learner, opt_name, nami...` | Named base rather than a zero-argument `super()`: `slots=True` rebuilds the class, and on Python below 3.12.4 -- inside the project floor -- the `__class__` cell still points at the discarded one, so `super()` raises "obj must be an instance or subtype of type" here. |
| 874 | `naming(name)` | Reserved with the rest, so an auto-named flow layer cannot claim the name afterwards. |
| 899 | `parts = cast(list[object], pattern.model_dump(by_alias=True))[1:]` | `nnx.Optimizer` requires `wrt`, and the parameters are the only sensible default. `Param` and `flax.nnx` itself are default imports of the learner, so nothing is added here. |

## `src/structcast_model/builders/keras.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 70 | `return None` | A lambda or another expression form: not a constructor call, so not a layer to judge. |
| 139 | `rate += ", keyword or first positional"` | The family takes its rate first and is the only one whose first argument is read as one, so the positional spelling is offered to it alone. |
| 175 | `prologue += f"{sep}structcast_model.keras.layers.disable_flash_attention_for_remat()"` | Before the sub-layers are built, not at training time: `keras.layers.MultiHeadAttention` caches the flash attention decision in its own `__init__`, so a later flip misses it. |
| 176 | `body = "_call_body"` | No base class: Keras reads the `call` signature to decide whether it forwards `training` and how it maps a batch passed by name, and a `*args` base would erase both. Not `_call_impl`: on the torch backend a Keras layer inherits `torch.nn.Module`, which owns that name for its call dispatcher, so it is not this emission's to take. |
| 177 | `remat = "keras.remat(lambda *arrays: self._call_body(*arrays, training=training, **kwar...` | The rematerialized callable takes the arrays positionally and reads the flags off the closure: on the TensorFlow backend the custom gradient behind `keras.remat` refuses keyword arguments outside eager execution, which is every compiled training step. |
| 475 | `elsewhere = {name for units, _ in self._segments for _, output, _ in units for name in ...` | A segment is one function the adapter calls with the batch alone, so a value another segment computed is simply not in scope there. |
| 513 | `closed = {"self", "kwargs", *self.models, *self.others, *self.layers}` | `_flow_<optimizer>` and `_flow_inference` are left out: only `__init__` reads them, and a batch parameter or a flow's local never lands in that scope. |
| 516 | `for name in unique([*self.models, *self.others, *self.layers, *self.inputs, *stored]):` | `__init__` binds these after the models and layers, so a model or layer under one of them is rebound before any flow runs. |
| 575 | `body.append(f'inners = [getattr(s.optimizer, "inner_optimizer", s.optimizer) for s in {...` | After `prepare`: under a float16 policy the accumulation window is the wrapped inner optimizer's, and the wrapping is final by now. |
| 576 | `body.append("windows = sorted({inner.gradient_accumulation_steps or 1 for inner in inne...` | The counters answer for the whole learner, so the optimizers must agree on one window (`docs/adr/0017`) -- a ValueError, since generated scripts import no builder errors. |
| 583 | `body.append('self._ema_optimizers = [inner for inner in inners if getattr(inner, "use_e...` | Off `inners`, which already reached through a float16 wrapper: `use_ema` belongs to the inner optimizer, the only one that keeps an average at all. |
| 591 | `clock = (` | The first segment is the learner's clock, read through the segment because `prepare` replaced the optimizer of a segment it wrapped for loss scaling (`docs/adr/0019`). One expression, no local: every name in a step's namespace is a batch input the user named. |
| 741 | `def _build_segment(  # noqa: PLR0913, PLR0917` | The base signature, narrowed to the Keras schema. |
| 752 | `base = BaseLearnerBuilder._build_segment(self, imports, module, learner, opt_name, nami...` | Named base rather than a zero-argument `super()`: `slots=True` rebuilds the class, and on Python below 3.12.4 -- inside the project floor -- the `__class__` cell still points at the discarded one, so `super()` raises here, exactly as in the Flax builder. |

## `src/structcast_model/builders/schema.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 286 | `TensorSpecTree = Annotated[` | mypy resolves the recursion only in this implicit alias form, while pydantic builds a recursive schema only from `TypeAliasType`, so the two forms of the same alias are kept side by side. |
| 530 | `target_type: ClassVar[type[SerializableT]] = cast(type[SerializableT], WithExtra)` | Subclasses bind `target_type` to the concrete type they parametrize `Template` with, which a `ClassVar` cannot express, so the default is cast to the type variable. Dropping `ClassVar` instead would make pydantic treat the attribute as a model field. |
| 576 | `if merged:` | `create` is annotated to return the base `Parameters` while it instantiates `cls` at runtime. |

## `src/structcast_model/builders/torch.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 63 | `keywords = ", ".join(f"{k!r}: {v}" for k, v in self.gradient_checkpointing.items())` | Always keyworded: `_resolve_gradient_checkpointing` fills `use_reentrant` in, so a checkpointed layer never carries an empty mapping. |
| 313 | `defs: list[str] = []` | Freezing restores each owned model's construction-time requires_grad states instead of a blanket True, so submodules the user froze stay frozen across optimizer segments. |
| 315 | `used: list[str] = list(self.models)` | `others` the step body reads off `self`, so the body lines stay plain local-variable code. |
| 321 | `scaled = f"({loss} / {self.accumulate_gradients})" if self.accumulate_gradients else loss` | Scale inside the backward expression so the reported loss keeps its unscaled value. |
| 327 | `params = ["__need_update__", *[n for n in info["external"] if n in available]]` | The gate leads the parameters of a training flow function: `_gated_body` reads it, and the inference flow, which gates nothing, takes no such parameter. |
| 361 | `binds = [` | Incrementing `_steps` first keeps `_steps` on the trainer's old 1-based clock, so the `(+ 1) % k` gate preserves the historically short first accumulation window. |
| 380 | `tail.append("if self._has_updated:")` | One blend per Update, never per accumulation micro-step, and after every segment of the step has applied: what an average follows is the weights a whole step produced. |
| 398 | `f'if any(type(p).__name__ == "DTensor" for p in {model}.parameters()):',` | One parameter walk covers both refusals: FSDP2 shards every parameter and forbids the copy outright, tensor parallelism shards only the modules its plan matched and then breaks on the first blend of the mixed list that leaves behind. The type name is all the check reads -- the generated learner cannot import `torch.distributed.tensor` for an `isinstance` -- and walking the parameters is what sees through a DDP or compile wrapper without naming either. |
| 586 | `clip = cast(TorchLearnerBehavior, learner).CLIP` | `learner` arrives through the base hook signature; `template_type` guarantees the torch schema. |
| 628 | `naming(name)` | Reserved with the rest, so an auto-named flow layer cannot claim the name afterwards. |
| 647 | `if isinstance(mixed_precision, bool) and not mixed_precision:` | The precision type is not read here: `_validate_mixed_precision` already refuses an enabled MIXED_PRECISION with anything but float16, so a scaler is built only where one is wanted. |

## `src/structcast_model/builders/utils.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 91 | `evaluated: list[str] = []` | Every `eval:` value rendered so far, so a binding can tell whether it read one of them. |
| 124 | `imports[f"{FILE_IMPORT_PREFIX}{first.file}"].add(first.address)` | File-addressed objects cannot be imported by module name: record the file under a special key so the script renderer emits an import_from_address binding instead. |
| 126 | `project = module.split(".")[0] in ("structcast", "structcast_model")` | The project's own packages are imported as modules and reached fully qualified, so none of their members becomes a global of the generated script. The address is emitted as written: a framework package re-exports symbols and its `layers` subpackage only, so `structcast_model.torch.create_opt` resolves while the module path `structcast_model.torch.optimizers.create_opt` fails when the script runs, by design. |
| 147 | `aname, kwname = f"_arg{bind_index}", f"_kw{bind_index}"` | The position in `rest` is deterministic, unlike an id()-derived suffix, so the same pattern always renders the same script. Reuse across nesting levels is safe: a nested lambda only ever references its own arguments, shadowing any outer ones. |
| 154 | `if len(evaluated) == seen and _MODULE_LEVEL_NAME.fullmatch(target) and _literal_argumen...` | A closure over nothing but constants and one module-level name is hoisted, so that every layer binding that callable to those arguments shares the one object: built per instance instead, it would be a per-instance leaf of the Flax graphdef, and two instances of the generated class would no longer hit the same `flax.nnx.jit` trace. An `eval:` value is written to be read where the object is built -- `rngs` is the standing example -- so a binding that read one stays where the reading works. |

## `src/structcast_model/commands/cmd_flax.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 13 | `import structcast_model as scm` | `scm`, `scm_flax` and `scm_loggers` are package shims routing to lazy submodules, so importing them pulls in no framework. Wrapping them in `LazyModuleImporter` would not work: it copies the shim's still unresolved submodule slots, so every access after the first would hand back `None`. |
| 181 | `os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(fraction)` | Preallocation is turned off alongside the fraction so the share is taken as the run needs it; `setdefault`, because an operator who set the variable deliberately keeps their choice. |
| 198 | `preset = cast('Literal["single", "dp", "fsdp", "tp", "fsdp_tp"]', strategy)` | Cast, not validate: the strategy owns the list of presets it knows and rejects the rest with the names it accepts, which is the error a mistyped preset should read. |
| 204 | `def train(  # noqa: PLR0913, PLR0917` | The CLI surface: every training option is one Typer parameter. |
| 292 | `_cap_gpu_memory(gpu_memory_fraction)` | Before the line below, which is the first `jax` attribute access of the run and therefore what imports JAX: the variables it writes are read once, while JAX brings up its backend. |
| 294 | `strategy = _resolve_strategy(strategy_pattern, device)` | First: constructing the strategy activates its mesh process-wide, and every array allocated afterwards -- the model parameters above all -- is placed against it. |
| 303 | `raise ValueError(` | The pattern already called the class, so the command holds a built module and the run's RNG never reached it. Calling it anyway would run the module's forward pass with a `rngs` keyword and fail somewhere inside the generated model instead. |
| 310 | `models = strategy.wrap(models)` | Before the learner: an optimizer inherits the sharding of the parameters it is built over, and the learner's inference views are taken from the models as they are when it is constructed. |
| 315 | `fixed = {"static_argnames", "static_argnums", "donate_argnames", "donate_argnums"}` | Both spellings of both contract arguments go: --help promises they cannot be overridden, and the argnums form would renumber what the argnames form fixes. |
| 318 | `step = getattr(learner, flow_name)` | The generated learner names its steps after the contract they follow: only the training one rewrites state, so only its parameters are donated (`docs/adr/0019`, amended by `docs/adr/0023`: a scaled step's loss scales are donated along with them). The inference step runs against views sharing the models' arrays and donates nothing. |
| 332 | `logger_type = scm_loggers.MLflowLogger if logger_name == "mlflow" else scm_loggers.Wand...` | Built before the resume, which fetches the state through it. Only the experiment name is stored here: the run itself starts in __enter__. |
| 339 | `models=dict(learner.models),` | The learner's mapping, not the command's: the saver writes `learner.models`, which also carries the `ema_<model>` shadows the command never built (`docs/adr/0021`). |
| 380 | `"mesh": dict(strategy.mesh.shape),` | A plain dict: the mesh reports its shape as an OrderedDict, which YAML tags as Python. |

## `src/structcast_model/commands/cmd_keras.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 16 | `import structcast_model as scm` | `scm`, `scm_keras` and `scm_loggers` are package shims routing to lazy submodules, so importing them pulls in no framework, as in cmd_flax. |
| 83 | `COMPILE_TAIL = (` | Both commands compile through the same backend adapter, so they say the same thing about it. The stateless scope is `time`'s alone to mention: it is a caveat about --training-mode, which only `time` has, and about numbers, which only `time` reports. |
| 204 | `print(f'Timing on the "{keras.backend.backend()}" Keras backend, device "{device}".')` | Unlike `train`, this command takes no --backend and inherits the ambient one (`docs/adr/0016`), so it says which one produced the number: the backend decides what actually executes. |
| 212 | `return model(inputs, training=training_mode)` | One keyword, not **batch: `create_numpy_inputs` answers with an array for a bare shape and a list for a sequence (keras/trainer.py), and only the mapping form could be splatted. |
| 219 | `with _compile_choice(keras.backend.backend(), compile_pattern):` | The seam `train` compiles through, so the number describes the step a run executes. The context manager is what refuses the torch backend, which builds no compiled step at all, and what puts the choice back afterwards: the adapter is one cached instance per process. |
| 254 | `if (active := keras.backend.backend()) != backend:` | The first attribute access is what imports Keras, hence after the assignment above. |
| 273 | `os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(fraction)` | Preallocation is turned off alongside the fraction so the share is taken as the run needs it; `setdefault`, because an operator who set the variable deliberately keeps their choice. |
| 300 | `enabled = raw if isinstance(raw, bool) else isinstance(raw, Mapping)` | The predicate of `keras.adapters.prepare`: any mapping enables the policy, an empty one included, so that a run is not loss-scaled by the adapter while computing in float32. |
| 320 | `preset = cast('Literal["single", "dp", "fsdp", "tp"]', strategy)` | Cast, not validate: the strategy owns the list of presets it knows, and which of them the active backend supports, and rejects the rest with the reason -- which is the error to read. |
| 377 | `def train(  # noqa: PLR0913, PLR0917` | The CLI surface: every training option is one Typer parameter. |
| 457 | `_cap_gpu_memory(backend, gpu_memory_fraction)` | Before the activation below, which is what imports Keras: the variables it writes are read once, while the backend's framework starts up. The torch cap is an API call, so it waits. |
| 461 | `with _compile_choice(backend, compile_pattern):` | Spanning everything that builds or traces a step: the learner's constructor, and `wrap_steps`. |
| 462 | `factory = instantiate_object(learner_pattern)` | Resolved before the models: the class itself is what carries the policy the models are built under, and instantiating it needs them. |
| 464 | `print(f'Setting the global mixed precision policy to "{policy}"...')` | A policy only reaches the layers built after it is set, and the learner receives models that are already built, so this is the last moment it can be set (`docs/adr/0016`). |
| 469 | `with strategy.activate():` | Everything a run allocates is built inside the activation: a JAX variable reads the active distribution while it is created, and a MirroredStrategy mirrors only what its scope encloses -- the models above all, and the optimizers the learner builds against their variables. |
| 470 | `keras.utils.set_random_seed(seed + strategy.data_rank)` | Inside the activation: data_rank reads 0 until the process group is joined, and each rank needs its own seed so replicas draw different dropout masks, as `scm torch train` does. |
| 476 | `declared.update(scm_keras.resolve_input_shapes(built) or {})` | Read before the trace: `initial_model` wraps a layer into a functional `keras.Model`, which carries none of the layer's attributes, so the shapes it was traced with would be unrecoverable afterwards and the run would record none. |
| 479 | `models = strategy.wrap(models)` | Before the learner: it captures the model objects it is handed, and its optimizers are built against their variables while it is constructed. |
| 481 | `input_shapes = reduce_dict(shapes) or json.loads(json.dumps(declared))` | A declared shape is a tuple, which `arguments.yaml` would record as a `!!python/tuple` tag no safe YAML loader reads back; the round-trip makes it the plain data `--shape` would have given. |
| 483 | `strategy.wrap_steps(learner)` | After the learner: the steps this rewires are the ones the backend adapter built in its constructor, and each one runs the replicas itself rather than being traced into a scope. |

## `src/structcast_model/commands/cmd_torch.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 14 | `import structcast_model as scm` | `scm`, `scm_loggers` and `scm_torch` are package shims routing to lazy submodules, so importing them pulls in no framework. Wrapping them in `LazyModuleImporter` would not work: it copies the shim's still unresolved submodule slots, so every access after the first would hand back `None`. |
| 60 | `shapes = scm_args.shapes_option(` | --shape and --device read differently under `train`, so the commands share only the prose that is true for both. |
| 264 | `torch.cuda.set_per_process_memory_fraction(fraction, torch.device(device).index)` | A bare "cuda" carries no index; torch then caps the current device, which is the same one. |
| 293 | `with torch.device(device):` | Everything below runs on the training device: the models, and the tracker buffers, which are allocated with torch.zeros and would otherwise fail the first step mixing CUDA criteria with CPU buffers. |
| 297 | `if is_main and resume is None:` | A resumed run loads its weights later, which would overwrite whatever the initializers and the initial-weight broadcast produce here. |
| 307 | `for optimizer in learner.optimizers.values():` | Before the resume reads or writes a single optimizer state: a group mixing DTensor and plain parameters crashes the first step under tensor parallelism, and a state saved from a split optimizer must load back into an identically split one. |
| 313 | `if not distributed:` | The flow functions are the compile units; the step itself stays eager. See ADR-0004. Flow functions compile only on a single device: distributed wrappers graph-break inside the flow, and the fragment overhead measurably exceeds the glue-fusion gain (H200 numbers in docs/references/flow-compile-step-time-h200.md). The models themselves compile either way. |
| 333 | `saver = scm_torch.TrainingStateSaver(logger=logger, strategy=strategy)` | The saver and the best-criterion monitors run collectives, so they are built on every rank; only rank 0 holds a real logger and writes anything. See ADR-0005. |
| 353 | `def train(  # noqa: PLR0913, PLR0917` | The CLI surface: every training option is one Typer parameter. |
| 443 | `_cap_gpu_memory(device, gpu_memory_fraction)` | After the resolution above, which is what decides the device the cap applies to, and before anything allocates on it. |
| 446 | `strategy = _resolve_strategy(strategy_pattern, device, local_rank, distributed)` | Before the seeding, which is derived from the strategy's data coordinates rather than the global rank: the ranks of one tensor-parallel group split a model, so they must draw the same dropout masks as each other and read the same slice of the dataset (ADR-0022). The coordinates go into the environment too, because a dataset is an independently instantiated object pattern the CLI hands nothing to, and a rank-aware loader has no other way to reach them. |
| 474 | `if is_main:` | Built before the resume, which fetches the state through it. Only the experiment name is stored here: the run itself starts in __enter__. |
| 483 | `models=dict(learner.models),` | The learner's mapping, not the command's: the saver writes `learner.models`, which also carries the `ema_<model>` shadows the command never built (`docs/adr/0021`). |
| 532 | `with logger:` | One path for every rank: the NullLogger ranks run the same lifecycle and discard it all. |
| 543 | `if hasattr(learner, "param_group_names"):` | Guarded because ``param_group_names`` is a torch-only extension the generated learner adds, not a member of the ``Learner`` protocol every learner here satisfies. |

## `src/structcast_model/commands/main.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 41 | `logging.getLogger().setLevel(log_level.upper())` | `basicConfig` only sets the level when it installs the handler, so anything that configured logging first (a library imported on the way, a test harness) would keep its own level. |
| 86 | `app()` | `python -m structcast_model.commands.main` (the documented torchrun launch) imports this module as __main__, which the lazy-import tail does not dispatch to the Typer app. |

## `src/structcast_model/commands/shared_args.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 34 | `PATH_FORM_HELP = " The pattern may also be given as a path to a YAML/JSON file holding ...` | The pattern options accepting a file path all say so the same way. |
| 244 | `training_mode = Option(` | torch and keras; flax carries its own nnx.view variant in cmd_flax.py |
| 251 | `learner_pattern = Option(` | The `train` options torch, flax and keras spell identically. |

## `src/structcast_model/flax/__init__.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 67 | `import_structure = {` | Each symbol is listed exactly once: _class_to_module is a dict comprehension, so a name listed twice silently keeps the last writer. A re-exported name goes under its defining module when that module has an entry of its own -- distributed re-exports get_jax_device, routed to utils instead. The layers subpackage stays submodule-only, as its torch twin does. |

## `src/structcast_model/flax/distributed.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 35 | `"tp": (),` | No default plan: which layers pair up into a column/row split is the model's own shape, so the tables carry no such rule and the strategy refuses to run without one. |
| 74 | `data = jax.random.key_data(live)` | A typed key is placed through its raw data: a key array's own sharding describes the physical uint32 array, whose rank is one higher than the key's. |
| 234 | `return jax.make_mesh(` | Named rather than left to `jax.make_mesh`, whose default axis type has changed between jax versions -- and the type is what decides whether a plain model traces at all. |
| 322 | `size = self._mesh.shape[AXIS]` | The data axis alone: on a two-dimensional mesh every device of one model axis group runs the same items, so a batch split by the whole mesh would be as many times too small. |
| 370 | `if name not in model_states:` | A state holding models the learner no longer has is ignored; one missing a model the learner does have is a resume the checkpoint cannot answer, not a `KeyError`. |
| 432 | `if layer is not None and getattr(layer, "dot_general", None) not in (None, jax.lax.dot_...` | `None` and `jax.lax.dot_general` are both "the default": nnx.Linear stores the function itself, nnx.LinearGeneral stores None and resolves it per call. |
| 436 | `batch = f"'{AXIS}'" if self.data_axis_mode == "explicit" else "None"` | An `out_sharding` may name Explicit axes only, so the batch dimension of the hook's spec is the data axis under an Explicit data axis and None under the Auto default. |
| 462 | `return self._model_spec(array, array.ndim - 1)` | The bias of a column-parallel layer splits with the kernel's output dimension, so a one-dimensional array is a candidate here where the other tactics leave it whole. |
| 464 | `return PartitionSpec() if array.ndim < 2 else self._model_spec(array, 0)` | The bias is pinned replicated, and the tactic is where that lives because a rule table cannot say it: a bias split along the model axis -- or added once per shard -- is counted as many times as the axis is wide by the reduction that follows, the one tensor-parallel mistake that reports a plausible loss instead of an error. |
| 494 | `__all__ = ["AXIS", "MODEL_AXIS", "PRESET_RULES", "TACTICS", "TP_PRESETS", "FlaxDistribu...` | The module constants are listed because the LazySelectedImporter tail below only exposes the names in `__all__`, and a caller naming a preset or writing a rule table reads them. |

## `src/structcast_model/flax/layers/checkpointing.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 37 | `arrays = (*args, *(kwargs.pop(name) for name in self.inputs[len(args) :] if name in kwa...` | A caller may pass the batch by name -- the CLI initializing a model does -- while the rematerialized callable takes the arrays positionally, so the declared inputs are moved. |

## `src/structcast_model/flax/layers/grn.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 47 | `norm = jnp.sqrt((x * x).sum(axis=self.reduction_axes, keepdims=True))` | `sqrt` has an infinite derivative at 0, so an all-zero channel back-propagates `inf * 0 = NaN`. Mirrors `optax.safe_norm`: mask that zero vector to ones and renorm, so the untaken branch stays finite and the zero-norm sub-gradient is 0, matching PyTorch `norm_backward` (the timm reference). See https://github.com/google-deepmind/optax/blob/main/optax/_src/numerics.py#L48-L83 and https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where on the inner `where`. |

## `src/structcast_model/flax/optimizers.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 61 | `unwrap_variables(optimizer.opt_state),` | The nnx `OptArray` wrappers defeat the filter unless the state is unwrapped first, and the filter itself is required because a scheduled inject state carries a second `learning_rate` entry (its schedule state) under `hyperparams_states`. |
| 161 | `carrier = jnp.where(finite, 0.0, jnp.nan)` | Differentiating `x * carrier` reproduces `carrier` as the gradient DynamicScale inspects, which is finite exactly when the real gradients are: the growth interval, the backoff and the floor then stay flax's own rather than a second copy of them here. |
| 167 | `after = gradient_steps(optimizer)` | Read before the rollback below, which would revert the count with the rest of the state: either read is None exactly when the transformation carries no window at all. |

## `src/structcast_model/flax/trainer.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 31 | `_logger = getLogger(__name__)` | `_logger`, not the usual `logger`: `restore_training_state` takes a `Logger` parameter named `logger`. |
| 366 | `setattr(` | A DynamicScale is immutable, so the restored one is bound back under the name the learner reported it under -- which is the attribute it keeps it in. |
| 388 | `learner.restore_counters(int(meta["step"]), int(meta["update"]))` | Seed the learner's counters from the meta, so the step, update and accumulation clocks continue where the saved run left off (docs/adr/0018). |

## `src/structcast_model/flax/utils.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 12 | `@lru_cache(maxsize=1)` | `jax.Device` is Any to mypy: jaxlib re-exports it from its `_jax` C extension, which ships no stubs. |
| 23 | `def get_jax_device(device: str \| None = None) -> jax.Device:  # type: ignore[no-any-uni...` | `jax.Device` is Any to mypy, as above. |

## `src/structcast_model/keras/__init__.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 76 | `import_structure = {` | Each symbol is listed exactly once: _class_to_module is a dict comprehension, so a name listed twice silently keeps the last writer. A re-exported name goes under its defining module when that module has an entry of its own -- distributed re-exports the utils helpers, routed to utils instead. The layers subpackage stays submodule-only, as its flax and torch twins do. |

## `src/structcast_model/keras/adapters.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 22 | `from typing_extensions import Protocol, runtime_checkable` | Protocol and runtime_checkable come from typing_extensions so that isinstance checks use inspect.getattr_static on Python 3.11 as well (backported from 3.12), as in base_trainer. |
| 34 | `jax = LazyModuleImporter("jax")` | An inactive backend's framework may not be installed, so each one is bound lazily and only resolved by the adapter the active backend selects, as in `loggers.state_backends`. |
| 201 | `enabled = mixed_precision if isinstance(mixed_precision, bool) else True` | Any mapping enables loss scaling, an empty one included: it carries the wrapper's keyword arguments, and `builders/torch.py` reads the same field the same way. |
| 204 | `raise ValueError(f"Optimizer segment {segment.name!r} has no trainable variables to upd...` | An optimizer built against no variable trains nothing and reports no error, the exact silent no-op docs/adr/0016 rejects the alternatives for. |
| 208 | `segment.optimizer.build(segment.variables)` | Building here rather than on the first update keeps every slot variable out of a compiled step, where TensorFlow forbids creating variables and JAX would trace them. |
| 253 | `scaled = segment.optimizer.scale_loss(loss)` | Outside `fit()` nobody scales the loss for us. `scale_loss` returns the loss untouched on an optimizer without a loss scale, so no branch is needed. |
| 289 | `fixed = {"static_argnames", "static_argnums", "donate_argnames", "donate_argnums"}` | Both spellings of both contract arguments go, as `cmd_flax` drops them for `nnx.jit`: one mapping is splatted into a training step and an inference step whose positional signatures differ, so a `donate_argnums` meant for the first would donate the second's live weights. |
| 304 | `pairs += zip(segment.optimizer.variables, optimizer_values, strict=True)` | The optimizer variables carry the loss scale `scale_loss` reads. |
| 321 | `updated = [scope.get_current_value(variable) for variable in state_variables]` | `mapping` seeds the scope with every state variable, so each one has a current value: the one the flow wrote, or the one threaded in. |
| 336 | `own, optimizer_values = segment.optimizer.stateless_apply(optimizers[index], grads, tra...` | `stateless_apply` opens its own scope, so it runs outside the flow's. |
| 344 | `def train_step(**batch: Any) -> dict[str, Any]:` | The batch is gathered back into one mapping for the jitted call: the state lists are the positional arguments the trace is built around, and a batch spread over keywords there would move with every learner's input names. |
| 351 | `_assign(state_variables, states)` | Assigning every step keeps the variables the single source of truth, so the tracker, the checkpoints and the next step all read what this step computed. |
| 415 | `for variable in segment.variables:` | Autograd accumulates into `.grad`, so a step that did not clear it would apply the sum of every step so far -- including what another segment's backward pass left behind on a shared variable. |
| 474 | `if not int(keras.ops.convert_to_numpy(optimizer.iterations)):` | The same host read `training_step` makes of this counter, on the same variable. |
| 476 | `for variable, average in zip(` | Paired positionally against the optimizer's own list, as `keras.callbacks.SwapEMAWeights` does: both are built from the variables `build` was given. |
| 541 | `_ADAPTERS: dict[str, Callable[[], BackendAdapter]] = {` | Typed as constructors rather than as classes so that a type checker verifies each one against the protocol here, where the mismatch is one line, instead of at the call site of a missing method. |

## `src/structcast_model/keras/distributed.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 48 | `tf = LazyModuleImporter("tensorflow")` | An inactive backend's framework may not be installed, so each one is bound lazily and only resolved inside the preset paths the active backend can reach, as in `loggers.state_backends`. |
| 83 | `"tp": (),` | No default plan: which layers pair up into a column/row split is the model's own shape, and unlike the Flax twin a Keras variable carries no annotation of its own to fall back on -- so the preset is refused without rules rather than replicating everything and reporting success. |
| 151 | `return variable._layout  # noqa: SLF001` | The base class reads it first too; a caller may pin a layout. |
| 249 | `if (reason := REJECTED.get((self.preset, self._backend))) is not None:` | After the rule table: a mistyped tactic is wrong on every backend, so it is reported as itself rather than as whatever the active backend happens to say about the preset. |
| 258 | `available = len(self._device_names(limit=False))` | Unlimited: `_device_names` applies the very count being validated, so counting its result would report a negative or oversized count as the machine's own device count. |
| 330 | `self._scope = self._mirrored.scope()` | The scope object is held, not dropped: it owns the variable-creator scope it entered, and letting it be collected tears that down early -- after which the models are built as plain, unmirrored variables and the first step fails inside `strategy.run`. |
| 415 | `for segment in getattr(learner, "__dict__", {}).values():` | The loss, not the gradients: it is the one value the strategy can reach from out here, the segment's flow being what the adapter differentiates, and dividing it by the replica count turns the optimizer's SUM all-reduce into the mean of the per-replica gradients. The criteria the flow reports beside it are untouched, and still reduced with `ReduceOp.MEAN` below. What is scaled here is exactly what the generated shape exposes: one `AdapterSegment` per instance attribute (`docs/adr/0019`). Segments kept any other way -- inside a list, behind `__slots__`, built on the fly -- are invisible to this scan, so a learner holding them like that has to scale its own loss. |
| 420 | `for name in flows:` | The generated learner's public steps stay eager: `training_step` owns the host counters and reads the optimizer counter back after the step (`docs/adr/0018`), neither of which can run inside the replicated graph, so the strategy wraps the inner flow steps instead. `flow_functions` is what a learner exposes for exactly this rebinding, as in `cmd_flax`. |
| 462 | `return dict(batch)` | torch and `single`: on torch the loader hands each rank its own slice, exactly as the torch training path's `DistributedSampler` does -- a strategy that split the batch here would hand every rank the same data and quietly train on a fraction of the dataset. |
| 540 | `with _local_batch_losses():` | Traced under :func:`_local_batch_losses`: a Keras loss class would otherwise hand back the replica's share of the global batch's loss, which `ReduceOp.MEAN` below cannot tell from the per-replica means every other criterion is. |
| 574 | `reduced = torch.as_tensor(value, dtype=torch.float32).detach().clone()` | Detached and copied: the value a step returns is still the one its graph produced, and an in-place all-reduce would rewrite it under the optimizer that just ran. |
| 645 | `if variable.trainable or "seed_generator" in variable.path:` | Only the statistics: a non-trainable variable is not necessarily one, and the two other kinds must not be averaged. A `keras.random.SeedGenerator` -- which every Dropout or random-augmentation layer holds -- keeps its RNG state in an integer variable, so the division below would raise on it, and an averaged RNG state would be meaningless anyway; the same goes for any other integer counter a layer keeps. |
| 744 | `__all__ = [` | The module constants are listed because the LazySelectedImporter tail below only exposes the names in `__all__`, and a caller naming a preset or writing a rule table reads them. |

## `src/structcast_model/keras/layers/checkpointing.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 30 | `if keras.backend.backend() != "jax" or keras.config.is_flash_attention_enabled() is False:` | `is False`, not falsy: the default and the enabled state are both `None` ("attempt it"), and only an explicit disable reads back as False -- which is also what keeps the warning to one line however many checkpointed layers a model builds. |

## `src/structcast_model/keras/layers/grn.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 79 | `norm = ops.sqrt(ops.sum(ops.square(inputs), axis=self.reduction_axes, keepdims=True))` | `sqrt` has an infinite derivative at 0, so an all-zero channel back-propagates `inf * 0 = NaN`. This mirrors `optax.safe_norm`: mask the zero vector to ones and renorm, so the untaken branch stays finite and the zero-norm sub-gradient is 0, matching PyTorch `norm_backward` (timm). See https://github.com/google-deepmind/optax/blob/main/optax/_src/numerics.py#L48-L83 and https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where on the inner `where`. |

## `src/structcast_model/keras/trainer.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 20 | `_logger = getLogger(__name__)` | `_logger`, not the usual `logger`: `restore_training_state` takes a `Logger` parameter named `logger`, as in the flax twin. |
| 195 | `self.sums = {criterion: keras.ops.zeros((), dtype="float32") for criterion in self.crit...` | float32 whatever the criteria are: a run under a float16 or bfloat16 policy would otherwise accumulate an epoch of steps in the reduced type and lose the small ones. |
| 209 | `value = keras.ops.stop_gradient(criteria[criterion])` | The torch backend hands criteria still attached to the autograd graph; detaching keeps the epoch sum from retaining every step's graph and lets `logs` reach numpy. |
| 308 | `learner.restore_counters(int(meta["step"]), int(meta["update"]))` | Seed the learner's counters from the meta, so the step, update and accumulation clocks continue where the saved run left off (docs/adr/0018). |
| 367 | `self.logger.log_state_dict(self.strategy.state_dict(dict(info.models))["models"], name)` | The models alone, as the torch and flax twins save: best-value weights are for inference, so they carry no optimizer state, no counters and no wrapper key. |
| 398 | `"backend": keras.backend.backend(),` | Load-bearing: normalization statistics and RNG trajectories are not verified equivalent across the Keras backends, so a resume refuses a mismatch (`docs/adr/0016`). |

## `src/structcast_model/keras/utils.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 15 | `torch = LazyModuleImporter("torch")` | torch is only touched when the active Keras backend is torch (`get_keras_device`); binding it lazily keeps tensorflow and jax runs from importing it, as in `loggers.state_backends`. |
| 21 | `devices = [f"gpu:{index}" for index in range(torch.cuda.device_count())]` | The torch backend ships no distribution hooks (`keras.distribution` functions die on a None backend module), so the device list comes from torch itself, in the same "gpu:N" / "cpu:N" spelling the other backends report. |
| 122 | `raw = value.read_value() if hasattr(value, "read_value") else value` | Through `read_value` where the backend variable has one: under `tf.distribute` an optimizer counter is a `MirroredVariable` aggregated ONLY_FIRST_REPLICA, which refuses to become an array at all ("object __array__ method not producing an array") and hands back its primary copy through this call. A plain TensorFlow variable reads the same way, and the JAX and torch backends have no such method. |

## `src/structcast_model/loggers/__init__.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 31 | `import_structure = {` | Public symbols only: base's private helpers stay reachable by importing the module itself. |

## `src/structcast_model/loggers/base.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 121 | `__all__ = ["Logger", "NullLogger", "_epoch_metrics", "_local_training_state"]` | `_epoch_metrics` and `_local_training_state` are listed because the LazySelectedImporter tail below only exposes the names in `__all__`, and the two logger backends import them from here. |

## `src/structcast_model/loggers/mlflow.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 96 | `states = sorted(path.glob(f"*{self.state_backend.suffix}"))` | A directory artifact is what `mlflow.pytorch.log_state_dict` used to write, and what a backend saving a directory would write next: take this backend's format first, then the torch-flavored `state_dict.pth` of the runs recorded before. |
| 105 | `return TorchStateBackend().load(legacy[0])` | `*.pth` is a torch pickle whatever this logger's backend is, so it is read as one. |

## `src/structcast_model/loggers/state_backends.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 25 | `np = LazyModuleImporter("numpy")` | Every third-party module is bound lazily, as in `loggers.base`: importing this module must not drag torch into a Flax run nor jax into a torch one. They are touched inside `save` and `load` only. The Keras backend needs no keras at all -- a Keras state is numpy on the way out. |
| 66 | `return torch.load(path, map_location="cpu", weights_only=True)` | `weights_only` because the reference is user input, and an unpickled checkpoint executes code. |
| 88 | `with TemporaryDirectory(ignore_cleanup_errors=True) as workspace:` | A failed save leaves orbax's own temporary directory behind, and cleaning that up raises `Directory not empty` from `__exit__`, replacing the failure that caused it. |
| 94 | `for member in sorted(checkpoint.rglob("*")):` | Sorted, relative names: the archive does not record where it was built. |
| 105 | `raise RuntimeError(` | The extraction filters landed in 3.11.4, below the interpreters the project floor admits. Without them `extractall` takes no `filter`, and the raw `TypeError` reads as a bug here rather than as the interpreter being too old to extract safely. |
| 113 | `archive.extractall(checkpoint, filter="data")` | The reference is user input: `filter="data"` is the stdlib guard refusing members that would be written outside the destination. |
| 115 | `restored = dict(checkpointer.restore(checkpoint))` | Naming no arguments is what makes orbax read each item's handler from the checkpoint's own metadata and raise on one it cannot resolve. It logs a warning per item on the way, which is noise from a supported path, not a failure. |
| 155 | `with np.load(path) as archive:` | `np.load` does not unpickle unless asked to, and it is not asked to: the reference this path is reached with is user input, as in the torch backend's `weights_only`. |
| 176 | `raise TypeError(` | `ml_dtypes` registers bfloat16 as a user dtype, which `numpy.save` stores as an opaque 2-byte void and reads back as one: the values would survive and the type would not. |

## `src/structcast_model/loggers/wandb.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 98 | `return self.state_backend.load(Path(directory) / filename)` | The download is deleted with the temporary directory, so it is read inside the block. |

## `src/structcast_model/torch/__init__.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 85 | `import_structure = {` | Each symbol is listed exactly once: _class_to_module is a dict comprehension, so a name listed twice silently keeps the last writer. A re-exported name goes under its defining module when that module has an entry of its own -- trainer re-exports initial_distributed_env, get_torch_device and get_torch_device_type, routed to distributed/utils instead. CriteriaTracker is defined in the layers subpackage, whose entry stays submodule-only, so trainer routes it. |

## `src/structcast_model/torch/distributed.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 35 | `with try_import() as _fsdp_imports:` | torch >= 2.6 ships the stable per-parameter sharding (FSDP2) API. |
| 38 | `with try_import() as _dcp_imports:` | torch >= 2.2; older builds admitted by the torch-cpu extra floor lack both. |
| 42 | `with try_import() as _tp_imports:` | torch >= 2.4 ships the DTensor tensor-parallel styles at this path. |
| 197 | `plain = type(parameter) in (torch.Tensor, torch.nn.Parameter)` | Plain by exact type rather than ``isinstance(parameter, DTensor)``: the public DTensor path only exists from torch 2.5, while the tensor-parallel API that produces the mixture ships in 2.4 -- and a plain parameter is always exactly a Tensor or a Parameter. |
| 338 | `post = getattr(super(), "__post_init__", None)` | `object` has no `__post_init__`, so the chain up ends here rather than at a bare super() call. |
| 358 | `states: dict[str, Any] = {"models": {n: _innermost_module(m).state_dict() for n, m in m...` | Saved from the module the wrappers hold, which is exactly what the fallback load path writes back into: stripping the prefixes off the wrapper's own keys instead would also strip a `module.` a model owns itself. |
| 386 | `holds_state = state is not None` | Whether this rank was handed the state: the model tensors stay rank-0-only and reach the others through `broadcast_from_rank0`, so only the rank holding them can say what is missing. |
| 391 | `if holds_state:` | Checked before anything is written: torch reports a model it was handed an empty state for as a process-group failure, and a wrapped one accepts it silently and keeps its construction weights. A state holding models the learner no longer has is simply ignored. |
| 400 | `for name, module in models.items():` | The old-torch fallback saved wrapper-free keys, so it must load into the innermost module of whatever wrappers the CLI applied. |
| 406 | `api.set_model_state_dict(module, model_states.get(name, {}), options=self._load_options...` | `.get`, because the ranks the tensors are broadcast to hold none of them: what the state must carry was checked above, on the rank that was handed it. |
| 679 | `if not stripped or stripped == "_orig_mod":` | The root (or its compile wrapper's inner module) is never a match: wrap shards it last unconditionally, and a catch-all pattern must not shard it twice. |
| 777 | `return ColwiseParallel(use_local_output=False)` | The attention shape: the projection's output stays a DTensor, so the head reshape that consumes it sees the sharded head count instead of the full one. |
| 946 | `dtypes: dict[str, Any] = {k: _DTYPES[v] for k, v in self.mp_policy.items()}` | Any-valued because a dict[str, dtype] unpacked as **kwargs is checked against every MixedPrecisionPolicy field, including the bool cast_forward_inputs no dtype ever fills. |
| 950 | `for name, model in models.items():` | Every model is validated before any is sharded: a tie violation surfacing halfway would leave the earlier models already irrecoverably sharded. |
| 953 | `for _, submodule in reversed(matched[name]):` | Reversed pre-order shards descendants before ancestors; the other order makes an ancestor claim its whole subtree and re-sharding the descendant then throws. |
| 955 | `for model in models.values():` | fully_shard shards in place and hands back the very module it was given; its declared FSDPModule return type is the runtime-injected mixin, which is not statically an nn.Module. |
| 1113 | `]` | Unlike the package's other modules, this one is NOT replaced by LazySelectedImporter: generated flow functions call sync_gate inside torch.compile'd regions, and dynamo introspects the function's module through sys.modules — the shim raises on dunders (`__class__`) and breaks tracing (InternalTorchDynamoError). A plain module traces cleanly. |

## `src/structcast_model/torch/layers/channel_shuffle.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 19 | `def forward(self, input: Tensor) -> Tensor:` | pylint: disable=redefined-builtin |

## `src/structcast_model/torch/layers/checkpointing.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 35 | `return torch.utils.checkpoint.checkpoint(` | Only the positional arguments are tracked for recomputation, so the keyword arguments are bound into the callable, as `transformers.GradientCheckpointingLayer` does. |

## `src/structcast_model/torch/layers/reinmax.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 33 | `one_hot_sample, logits, y_soft, tau = cast("tuple[Tensor, Tensor, Tensor, Tensor]", ctx...` | `torch._C._FunctionBase.saved_tensors` is typed as a 1-tuple upstream, but it holds every saved tensor. |

## `src/structcast_model/torch/layers/split.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 26 | `def forward(self, input: Tensor) -> tuple[Tensor, ...]:` | pylint: disable=redefined-builtin |

## `src/structcast_model/torch/optimizers.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 74 | `pgs[group_name] = {"lr_scale": this_scale, "weight_decay": this_decay, "params": [], "p...` | "lr_scale" only works with timm schedulers |

## `src/structcast_model/torch/trainer.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 21 | `_logger = getLogger(__name__)` | `_logger`, not the usual `logger`: `restore_training_state` takes a `Logger` parameter named `logger`. |
| 178 | `tracker = cast("CriteriaTracker", compile_fn(tracker))` | torch.compile returns an OptimizedModule proxying the tracker, typed as a plain Module. |
| 245 | `self.logger.log_state_dict(self.strategy.state_dict(dict(info.models))["models"], name)` | Producing the states is a collective, so every rank must reach it. That the ranks agree on whether this epoch is the best is guaranteed by the tracker values being all-reduced. |
| 261 | `states = self.strategy.state_dict(dict(info.models), learner.optimizers, learner.optimi...` | Producing the states is a collective: every rank runs it, the null-logger ranks discard it. |
| 303 | `learner.restore_counters(int(meta["step"]), int(meta["update"]))` | Seed the learner's counters from the meta, so the step, update and accumulation clocks continue where the saved run left off (docs/adr/0018). |

## `src/structcast_model/utils/base.py`

| Line | Code | Reason |
| ---: | --- | --- |
| 89 | `value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)` | Handle the sequence of uppercase letters followed by a lowercase letter |
| 90 | `value = re.sub(r"([a-z])([A-Z])", r"\1_\2", value)` | Insert an underscore between a lowercase letter and an uppercase letter |
| 91 | `value = re.sub(r"([0-9])([A-Z])", r"\1_\2", value)` | Insert an underscore between a digit and an uppercase letter |
| 92 | `value = re.sub(r"([a-z])([0-9])", r"\1_\2", value)` | Insert an underscore between a lowercase letter and a digit |
