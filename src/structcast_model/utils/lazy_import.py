"""Utilities to import modules and classes.

The implementation is based on the following:
    - [lazy-imports](https://github.com/telekom/lazy-imports/tree/main).
    - [optuna](https://github.com/optuna/optuna/tree/master)
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType, TracebackType
from typing import Any


class LazySelectedImporter(ModuleType):
    """Do lazy imports."""

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: dict[str, list[str]],
        extra_objects: dict[str, Any] | None = None,
        skip_modules: list[str] | None = None,
    ):
        """Initialize a lazy selected importer.

        Args:
            name: Name of the module.
            module_file: Path to the module file.
            import_structure: Dictionary of the import structure.
            extra_objects: Extra objects to add to the module.
            skip_modules: Modules to skip.
        """
        super().__init__(name)
        skip_modules = skip_modules or []
        self._modules = {k for k in import_structure if k not in skip_modules}
        self._class_to_module = {v: k for k, values in import_structure.items() for v in values}
        self._objects = extra_objects or {}
        self._import_structure = import_structure
        # Needed for autocompletion in an IDE
        self.__all__ = list(self._modules) + sum(import_structure.values(), []) + list(self._objects)
        self.__file__ = module_file
        self.__path__ = [os.path.dirname(module_file)]

    # Needed for autocompletion in an IDE
    def __dir__(self) -> list[str]:
        """Return the directory.

        Returns:
            The directory.
        """
        exclude = {"_modules", "_class_to_module", "_objects", "_name", "_import_structure"}
        return list(set(super().__dir__()) - exclude) + self.__all__

    def __getattr__(self, name: str) -> Any:
        """Get an attribute.

        Args:
            name: Name of the attribute.

        Returns:
            The attribute.
        """
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module:
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str) -> ModuleType:
        """Get a module.

        Args:
            module_name: Name of the module.

        Returns:
            The module.
        """
        return importlib.import_module("." + module_name, self.__name__)

    def __reduce__(self) -> tuple[type[LazySelectedImporter], tuple[str, str | None, dict[str, list[str]]]]:
        """Reduce the module.

        Returns:
            The Type of the module and the arguments.

        """
        return self.__class__, (self.__name__, self.__file__, self._import_structure)


class LazyModuleImporter(ModuleType):
    """Module wrapper for lazy import.

    This class wraps the specified modules and lazily imports them only when accessed.
    Otherwise, `import optuna` is slowed down by importing all submodules and
    dependencies even if not required.
    Within this project's usage, importlib override this module's attribute on the first
    access and the imported submodule is directly accessed from the second access.

    Args:
        name: Name of module to apply lazy import.
    """

    def __init__(self, name: str) -> None:
        """Initialize a lazy module importer."""
        super().__init__(name)
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        """Load the module.

        Returns:
            The module.
        """
        if self._module is None:
            self._module = importlib.import_module(self.__name__)
            self.__dict__.update(self._module.__dict__)
        return self._module

    def __getattr__(self, item: str) -> Any:
        """Get an attribute.

        Args:
            item: Name of the attribute.

        Returns:
            The attribute.
        """
        return getattr(self._load(), item)


class _DeferredImportExceptionContextManager:
    """Context manager to defer exceptions from imports.

    Catches :exc:`ImportError` and :exc:`SyntaxError`.
    If any exception is caught, this class raises an :exc:`ImportError` when being checked.

    """

    def __init__(self) -> None:
        """Initialize a deferred import exception context manager."""
        self._deferred: tuple[Exception, str] | None = None

    def __enter__(self) -> _DeferredImportExceptionContextManager:
        """Enter the context manager.

        Returns:
            Itself.
        """
        return self

    def __exit__(
        self,
        exc_type: type[Exception] | None,
        exc_value: Exception | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Exit the context manager.

        Args:
            exc_type:
                Raised exception type. :obj:`None` if nothing is raised.
            exc_value:
                Raised exception object. :obj:`None` if nothing is raised.
            traceback:
                Associated traceback. :obj:`None` if nothing is raised.

        Returns:
            :obj:`True` if the exception is caught, :obj:`None` otherwise.

        """
        if isinstance(exc_value, (ImportError, SyntaxError)):
            if isinstance(exc_value, ImportError):
                message = (
                    f"Tried to import '{exc_value.name}' but failed. Please make sure that the package is "
                    f"installed correctly to use this feature. Actual error: {exc_value}."
                )
            elif isinstance(exc_value, SyntaxError):
                message = (
                    f"Tried to import a package but failed due to a syntax error in {exc_value.filename}. "
                    f"Please make sure that the Python version is correct to use this feature."
                    f" Actual error: {exc_value}."
                )
            else:
                raise exc_value

            self._deferred = (exc_value, message)
            return True
        return None

    @property
    def is_successful(self) -> bool:
        """Return whether the context manager has caught any exceptions.

        Returns:
            :obj:`True` if no exceptions are caught, :obj:`False` otherwise.

        """
        return self._deferred is None

    def check(self) -> None:
        """Check whether the context manger has caught any exceptions.

        Raises:
            ImportError: If any exceptions are caught.
        """
        if self._deferred is not None:
            exc_value, message = self._deferred
            raise ImportError(message) from exc_value


def try_import() -> _DeferredImportExceptionContextManager:
    """Create a context manager that can wrap imports of optional packages to defer exceptions.

    Returns:
        Deferred import context manager.
    """
    return _DeferredImportExceptionContextManager()
