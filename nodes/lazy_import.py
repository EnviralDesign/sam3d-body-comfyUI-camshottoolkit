from __future__ import annotations

from importlib import import_module


class LazyModule:
    """Import a module only when one of its attributes is first used."""

    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = import_module(self._module_name)
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)
