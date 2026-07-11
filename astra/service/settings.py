"""Configuration for the ASTRA reference service, read from environment variables.

No pydantic-settings dependency: a plain dataclass plus a :meth:`ServiceSettings.from_env`
classmethod. Every knob has an ``ASTRA_*`` environment variable:

=====================  ==========================  =====================================
Env var                Default                     Meaning
=====================  ==========================  =====================================
ASTRA_CONFIG           ``None`` (pipeline default  Config YAML governing data prep and
                       ``configs/defaults.yaml``)  supplying the default model name
                                                   (config-first, like the training CLI).
ASTRA_MODEL_NAME       ``model_name`` from         Name the model was trained/exported
                       ASTRA_CONFIG or             under (``AstraPredictor.load``).
                       configs/defaults.yaml
                       (``None`` if unreadable)
ASTRA_ARTIFACTS_DIR    ``models``                  Root artifacts dir (bundle, weights).
ASTRA_DATA_DIR         ``data/raw``                Population CSV directory.
ASTRA_PATIENT_DIR      ``data/patients``           Pre-split per-patient CSV directory.
ASTRA_DEVICE           ``None`` (auto-detect)      'cuda' | 'cpu'.
ASTRA_CACHE_SIZE       ``8``                       Patient contexts kept warm (LRU).
ASTRA_LOG_LEVEL        ``INFO``                    Console log level name.
=====================  ==========================  =====================================
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = "models"
DEFAULT_DATA_DIR = "data/raw"
DEFAULT_PATIENT_DIR = "data/patients"
DEFAULT_CACHE_SIZE = 8
DEFAULT_LOG_LEVEL = "INFO"


def _default_model_name(config_path: Optional[str] = None) -> Optional[str]:
    """``model_name`` from *config_path* (or ``configs/defaults.yaml``) —
    ``None`` if unavailable.

    Imported lazily so machines without the repo config (or with a broken
    ``astra.utils`` import) can still construct settings; the caller decides
    whether a missing model name is fatal.
    """
    try:
        from astra.utils import get_cfg

        return get_cfg(config_path).get("model_name")
    except Exception:  # noqa: BLE001 — any config failure just means "no default"
        logger.warning(
            "Could not read default model_name from %s; "
            "set ASTRA_MODEL_NAME explicitly.",
            config_path or "configs/defaults.yaml",
            exc_info=True,
        )
        return None


@dataclass
class ServiceSettings:
    """Runtime configuration for :func:`astra.service.app.create_app`."""

    model_name: Optional[str] = None
    config_path: Optional[str] = None
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR
    data_dir: str = DEFAULT_DATA_DIR
    patient_dir: str = DEFAULT_PATIENT_DIR
    device: Optional[str] = None
    cache_size: int = DEFAULT_CACHE_SIZE
    log_level: str = DEFAULT_LOG_LEVEL

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        """Build settings from ``ASTRA_*`` environment variables.

        Empty-string values are treated as unset. ``ASTRA_MODEL_NAME`` falls
        back to the ``ASTRA_CONFIG`` config (or the repo default config),
        lazily and tolerating failure -> ``None``.
        """
        config_path = os.environ.get("ASTRA_CONFIG") or None
        model_name = (os.environ.get("ASTRA_MODEL_NAME")
                      or _default_model_name(config_path))

        cache_size = DEFAULT_CACHE_SIZE
        raw_cache = os.environ.get("ASTRA_CACHE_SIZE")
        if raw_cache:
            try:
                cache_size = int(raw_cache)
            except ValueError:
                logger.warning(
                    "Invalid ASTRA_CACHE_SIZE=%r — using default %d",
                    raw_cache, DEFAULT_CACHE_SIZE,
                )

        return cls(
            model_name=model_name,
            config_path=config_path,
            artifacts_dir=os.environ.get("ASTRA_ARTIFACTS_DIR") or DEFAULT_ARTIFACTS_DIR,
            data_dir=os.environ.get("ASTRA_DATA_DIR") or DEFAULT_DATA_DIR,
            patient_dir=os.environ.get("ASTRA_PATIENT_DIR") or DEFAULT_PATIENT_DIR,
            device=os.environ.get("ASTRA_DEVICE") or None,
            cache_size=cache_size,
            log_level=(os.environ.get("ASTRA_LOG_LEVEL") or DEFAULT_LOG_LEVEL).upper(),
        )

    @property
    def log_level_int(self) -> int:
        """Numeric logging level for ``setup_logging`` (unknown names -> INFO)."""
        level = logging.getLevelName(self.log_level.upper())
        return level if isinstance(level, int) else logging.INFO
