"""Patient data source abstraction for inference.

The inference pipeline reads raw per-patient EHR data (concept tables such as
``VitaleVaerdier``, ``Labsvar``, ``ADTHaendelser`` …) through a single seam:
:func:`astra.inference.patient_store.load_patient_csv`. By default that reads
CSV files. Deployments that obtain data differently (SQL queries, parquet, a
message bus) implement :class:`PatientDataSource` and register it with
:func:`astra.inference.patient_store.set_data_source` — every downstream step
(EWS→vitals merge, notes-derived features, Elixhauser scoring, ADT trajectory
building, binning, normalization) then runs unchanged on top of it.

The normative contract is the **DataFrame schema per concept**, not a file
format — see ``docs/HANDOFF.md``. Universal requirements:

- A ``CPR_hash`` column of dtype ``str`` (rows may cover only that patient;
  the pipeline re-filters defensively).
- Timestamp columns parseable by ``pd.to_datetime(..., format="mixed")``;
  which column is the timestamp per concept is defined in
  ``data/external/metadata.csv``.
- No CSV index-column artifacts (return clean frames; ``index_col``-style
  handling only applies to the file-based path).
"""

import logging
from typing import Dict, Optional, Protocol, runtime_checkable

import pandas as pd

logger = logging.getLogger(__name__)


@runtime_checkable
class PatientDataSource(Protocol):
    """Provides raw per-patient concept data to the inference pipeline.

    Implementations may also expose an optional capability::

        def fetch_prehospital(self, cpr_hash: str) -> Optional[dict]

    returning ``{'prehospital_start': timestamp-like, 'A': …, 'B': …,
    'C': …, 'D': …}`` (ABCD assessment values optional) or ``None`` when the
    patient has no prehospital record. When absent or returning ``None``, the
    patient's timeline is anchored at hospital arrival.
    """

    def fetch(self, cpr_hash: str, concept: str) -> Optional[pd.DataFrame]:
        """Return raw rows for one patient and concept, or ``None`` if the
        concept has no data for this patient (or is not served at all)."""
        ...


class CSVDataSource:
    """Reference adapter: reads concept CSVs from disk.

    Mirrors the historical file behavior of ``load_patient_csv``: a pre-split
    per-patient file ``{patient_dir}/{cpr_hash}/{concept}.csv`` is preferred,
    falling back to the population file ``{data_dir}/{concept}.csv`` filtered
    by ``CPR_hash``.

    Note: :class:`~astra.inference.api.AstraPredictor` treats this class as a
    directory configuration and keeps using the built-in file path (which
    honors per-call ``read_csv`` kwargs exactly); registering it via
    ``set_data_source`` also works but applies uniform read settings.
    """

    def __init__(self, data_dir: str = 'data/raw',
                 patient_dir: str = 'data/patients', cache: bool = True):
        self.data_dir = data_dir
        self.patient_dir = patient_dir
        self.cache = cache

    def fetch(self, cpr_hash: str, concept: str) -> Optional[pd.DataFrame]:
        from astra.inference.patient_store import load_patient_csv
        try:
            return load_patient_csv(
                cpr_hash, concept,
                data_dir=self.data_dir, patient_dir=self.patient_dir,
                cache=self.cache, low_memory=False, index_col=0,
                _bypass_source=True,
            )
        except FileNotFoundError:
            return None

    def __repr__(self):
        return (f"CSVDataSource(data_dir={self.data_dir!r}, "
                f"patient_dir={self.patient_dir!r})")


class InMemoryDataSource:
    """Serves pre-fetched DataFrames from memory.

    Useful for tests and for deployments that batch-query all concepts for a
    patient up front::

        source = InMemoryDataSource({
            cpr_hash: {'VitaleVaerdier': vitals_df, 'Labsvar': labs_df, ...},
        }, prehospital={cpr_hash: {'prehospital_start': ts, 'A': 'Fri', ...}})
    """

    def __init__(self,
                 patients: Dict[str, Dict[str, pd.DataFrame]],
                 prehospital: Optional[Dict[str, dict]] = None):
        self._patients = patients
        self._prehospital = prehospital or {}

    def fetch(self, cpr_hash: str, concept: str) -> Optional[pd.DataFrame]:
        df = self._patients.get(cpr_hash, {}).get(concept)
        if df is None:
            logger.debug("InMemoryDataSource: no %s for %s", concept, cpr_hash)
            return None
        return df.copy()

    def fetch_prehospital(self, cpr_hash: str) -> Optional[dict]:
        rec = self._prehospital.get(cpr_hash)
        return dict(rec) if rec is not None else None

    def __repr__(self):
        return f"InMemoryDataSource(n_patients={len(self._patients)})"
