# Cross-framework CLI declarations live in `commands/shared_args.py`

> **Amended by ADR-0024.** Keras `--compile` is graph compilation like the others, so it is no
> longer the third rule's example, and all six declarations (`torch|flax|keras` × `time|train`) come
> from one factory, `compile_option(api, tail)` — rule two's compile text, which named only torch
> and flax, now covers every command. Both rules below are edited to match.

The torch, keras and flax sub-apps declare largely parallel Typer options, and the copies drift: one
round of parallel edits left the same "path form" sentence spelled three ways across the three
`model_pattern` declarations, and the `--compile` skeleton differing between torch and flax for no
semantic reason. Short flags were separately made globally unique ("one letter, one meaning" — `-c`
is always `--compile`, `-s` always `--shape`; `--classname` moved to `-n`, `--sublayer` lost its
short flag rather than take `-S`, one shift-key away from the collision it removes).

We decided to split the commands package into a declaration layer and a functional layer.
`commands/shared_args.py` holds the shared Typer `Option`/`Argument` declarations and every
help-text builder (`TEMPLATE_PARAM_HELP`, `object_pattern_help`, the shapes/compile factories);
`commands/utils.py` keeps runtime functionality only (the value parsers, `reduce_dict`,
`instantiate_object`). Placement follows three rules:

- **Identical in two or more frameworks → one shared instance.** Sharing a single
  `OptionInfo`/`ArgumentInfo` across commands and apps is safe: typer copies the info object per
  registered parameter (`copy(default)` in `typer/utils.py`) and its param-building path is
  read-only (verified against typer 0.27.1, including a caveat list: the shallow copy shares the
  `parser` callable, so parsers must stay stateless, and each signature still needs its own correct
  annotation). A pair qualifies — `training_mode` is shared by torch and keras while flax keeps its
  own `nnx.view` variant next to the code that needs it.
- **Near-identical → a factory with the differences as explicit arguments.** The shapes and the
  compile texts share one skeleton with framework-specific slots (example shape order, initializer
  module, compile API, and the sentence each command adds after it). Drift now has to get past a
  function signature instead of a copy-paste.
- **Framework-specific semantics stay in `cmd_*.py`.** `--device` means three different things, and
  `--strategy` takes a preset name in flax and only a pattern in torch; centralizing them would buy
  no drift protection and cost locality.

"One letter, one meaning" is enforced by a test that walks the whole app tree and asserts every
short flag maps to exactly one long option, so the convention survives the planned keras/flax
`create learner` additions without relying on review memory.

Considered and rejected: centralizing every module-level declaration regardless of semantics
(reads well in one file but the divergent options gain nothing and every `cmd_*.py` reader pays a
file hop); sharing only when all three frameworks agree (leaves the torch/keras `training_mode`
pair duplicated — the only observed drift was exactly this class of copy); and keeping the help
constants in `utils.py` (two homes for declaration text is a second drift source).
