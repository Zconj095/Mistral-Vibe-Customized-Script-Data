from __future__ import annotations

__all__ = ["__version__", "run_programmatic"]
__version__ = "1.1.3"


def __getattr__(name: str):
    if name == "run_programmatic":
        from vibe.core.programmatic import run_programmatic

        return run_programmatic
    raise AttributeError(f"module 'vibe.core' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
