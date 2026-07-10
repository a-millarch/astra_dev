"""ASTRA reference inference service (FastAPI).

A thin HTTP wrapper around :class:`astra.inference.api.AstraPredictor` for
external teams. Single-worker, no auth/TLS — see :mod:`astra.service.app`.

Usage::

    pip install 'astra[service]'
    python -m astra.service --host 0.0.0.0 --port 8000

``create_app`` is imported lazily so that ``import astra.service`` (and
``ServiceSettings``) work without FastAPI installed.
"""

from astra.service.settings import ServiceSettings

__all__ = ["ServiceSettings", "create_app"]


def __getattr__(name):
    if name == "create_app":
        from astra.service.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
