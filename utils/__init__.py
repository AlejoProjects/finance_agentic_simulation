from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "data_visualization",
    "simulation_utils",
    "simulation_analysis",
    "personalities",
    "local_models",
    "apis",
    "api_keys",
    "fcla",
]


def __getattr__(name: str) -> Any:
    """This function lazily imports a requested utility module.

    Params:
        name: Requested module or simulation name.
    """
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
