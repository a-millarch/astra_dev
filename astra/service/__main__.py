"""Run the ASTRA reference service: ``python -m astra.service [--host H --port P]``.

Starts a single uvicorn worker (the service is single-worker by design — see
:mod:`astra.service.app`). Configuration comes from ``ASTRA_*`` environment
variables (:class:`astra.service.settings.ServiceSettings`).
"""

import argparse


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m astra.service",
        description="ASTRA reference inference service (single-worker uvicorn).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover - depends on install extras
        raise SystemExit(
            "uvicorn is not installed. Install the service dependencies first:\n"
            "    pip install 'astra[service]'   (or: pip install fastapi uvicorn httpx)"
        ) from e

    from astra.service.app import create_app
    from astra.service.settings import ServiceSettings

    app = create_app(settings=ServiceSettings.from_env())
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
