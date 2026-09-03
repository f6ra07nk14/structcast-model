"""Constants shared by builder modules."""

FILE_IMPORT_PREFIX = "__file__:"
"""Prefix marking a collected import that refers to a file path instead of a module name."""

BOUND_CALLABLE_PREFIX = "__bound__:"
"""Prefix marking a collected entry that is a module-level bound callable, not an import.

The rest of the key is the name the generated module binds it under, and the value carries the one
expression it is bound to."""
