# mypy resolves tests against src, with site-package types and Any tripwires

`mypy tests` used to run with `no_site_packages = true` and no `mypy_path`: the installed
`structcast_model` package was invisible, so `ignore_missing_imports = true` silently degraded every
import from it — including all Protocol classes — to `Any`, and structural conformance in tests was
never checked (issue #25: non-conforming Learner fakes surfaced only as runtime `AttributeError`s in
pytest). We decided: `mypy_path = ["src"]` so tests always resolve the package from source; drop
`no_site_packages` so typed dependencies (torch ships `py.typed`) participate in checking; and add
`disallow_any_unimported` + `disallow_subclassing_any` so a future resolution failure is a loud error
instead of silent `Any`. A canary test asserts mypy still rejects a non-conforming Learner, so a
config regression turns red instead of silently green.

Untyped dependencies are handled by escalation, narrowest mechanism first: an isolated boundary site
uses an explained `# type: ignore[no-any-unimported]` or an explicit `Any` (tqdm, jax); the
keras-facing modules get the two tripwires switched off via a scoped override (keras ships no
`py.typed` and its internals are not analyzable); and the in-house `structcast` package — annotated
but not yet shipping `py.typed` — is analyzed from its installed sources via
`follow_untyped_imports`, turning its formerly-`Any` API into real checking.

Considered and rejected: the split-import stopgap proposed in issue #25 (`typing_extensions` behind
`TYPE_CHECKING`) is a no-op — mypy bundles `typing_extensions` stubs in typeshed's stdlib, so
`no_site_packages` never hid them; and replacing global `ignore_missing_imports` with a per-module
missing-import allowlist was judged list-maintenance for little gain over the tripwires.
