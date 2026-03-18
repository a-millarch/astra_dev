"""
Per-patient CSV store for fast inference data loading.

Population-level CSVs (data/raw/) contain all ~13,000 patients and can be
>1 GB each.  Loading them to extract a single patient discards 99.99% of
rows — the dominant bottleneck in SimulationRunner (~294s of ~300s total).

This module provides:

1. ``load_patient_csv()`` — tries a pre-split per-patient file first,
   falls back to the full population CSV with CPR_hash filtering.

2. ``split_population_csvs()`` — one-time utility that splits population
   CSVs into ``data/patients/{cpr_hash}/`` directories.

Usage::

    # One-time pre-split (run once, or via ``make split_patients``)
    python -m astra.inference.patient_store --data-dir data/raw

    # In inference code — transparent fallback
    from astra.inference.patient_store import load_patient_csv
    df = load_patient_csv(cpr_hash, 'VitaleVaerdier', data_dir='data/raw')
"""

import os
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# All concept CSVs + supporting files that should be split per-patient.
# metadata.csv is excluded (small config file, not per-patient).
DEFAULT_SPLIT_FILENAMES = [
    'ADTHaendelser',
    'PatientInfo',
    'VitaleVaerdier',
    'Labsvar',
    'Medicin',
    'Procedurer',
    'ITAOversigtsrapport',
    'EWS',
    'Notater',
    'Diagnoser',
    'Cases',
    'IndUd',
    'ITARespirator',
]


def load_patient_csv(
    cpr_hash: str,
    filename: str,
    data_dir: str = 'data/raw',
    patient_dir: str = 'data/patients',
    cache: bool = True,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a CSV filtered to a single patient.

    Checks ``{patient_dir}/{cpr_hash}/{filename}.csv`` first.  If the
    per-patient file exists it is read directly (typically KB-sized).
    Otherwise falls back to the full population CSV at
    ``{data_dir}/{filename}.csv`` and filters by ``CPR_hash``.

    When *cache* is True (default), a fallback load automatically saves the
    filtered result to the per-patient directory so subsequent loads are fast.

    Args:
        cpr_hash: Patient identifier hash.
        filename: CSV stem without extension (e.g. ``'VitaleVaerdier'``).
        data_dir: Directory containing population-level CSVs.
        patient_dir: Root directory for per-patient subdirectories.
        cache: If True, save filtered result to per-patient dir on fallback.
        **read_csv_kwargs: Passed to ``pd.read_csv()`` (e.g. ``index_col``,
            ``low_memory``, ``dtype``).

    Returns:
        DataFrame containing only rows for *cpr_hash*.

    Raises:
        FileNotFoundError: If neither per-patient nor population CSV exists.
    """
    patient_path = os.path.join(patient_dir, cpr_hash, f'{filename}.csv')

    if os.path.isfile(patient_path):
        logger.debug("load_patient_csv: per-patient file %s", patient_path)
        return pd.read_csv(patient_path, **read_csv_kwargs)

    # Fallback: full population CSV + filter
    population_path = os.path.join(data_dir, f'{filename}.csv')
    logger.debug("load_patient_csv: fallback to population %s", population_path)
    df = pd.read_csv(population_path, **read_csv_kwargs)

    if 'CPR_hash' in df.columns:
        df = df[df['CPR_hash'] == cpr_hash]

    # Cache the filtered result for next time
    if cache and not df.empty:
        try:
            os.makedirs(os.path.dirname(patient_path), exist_ok=True)
            df.to_csv(patient_path, index=True)
            logger.info("load_patient_csv: cached %s (%d rows)", patient_path, len(df))
        except OSError as e:
            logger.warning("load_patient_csv: failed to cache %s: %s", patient_path, e)

    return df


def split_population_csvs(
    data_dir: str = 'data/raw',
    patient_dir: str = 'data/patients',
    cpr_hashes=None,
    filenames=None,
):
    """Split population-level CSVs into per-patient directories.

    For each source CSV, loads it once, groups by ``CPR_hash``, and writes
    one small CSV per patient to ``{patient_dir}/{cpr_hash}/{filename}.csv``.

    Processes one source CSV at a time to stay memory-safe for >1 GB files.
    Idempotent: overwrites existing per-patient files.

    Args:
        data_dir: Directory containing population-level CSVs.
        patient_dir: Root output directory for per-patient subdirectories.
        cpr_hashes: Optional set/list of CPR hashes to include. If None,
            discovers all unique hashes from PatientInfo.csv.
        filenames: Optional list of CSV stems to split. If None, uses
            :data:`DEFAULT_SPLIT_FILENAMES`.
    """
    if filenames is None:
        filenames = DEFAULT_SPLIT_FILENAMES

    # Discover all CPR hashes if not provided
    if cpr_hashes is None:
        pi_path = os.path.join(data_dir, 'PatientInfo.csv')
        logger.info("Discovering CPR hashes from %s", pi_path)
        pi = pd.read_csv(pi_path, usecols=['CPR_hash'], dtype={'CPR_hash': str})
        cpr_hashes = set(pi['CPR_hash'].dropna().unique())
        logger.info("Found %d unique CPR hashes", len(cpr_hashes))
    else:
        cpr_hashes = set(cpr_hashes)

    os.makedirs(patient_dir, exist_ok=True)

    for filename in filenames:
        src_path = os.path.join(data_dir, f'{filename}.csv')
        if not os.path.isfile(src_path):
            logger.info("Skipping %s (not found)", src_path)
            continue

        logger.info("Splitting %s ...", src_path)
        df = pd.read_csv(src_path, low_memory=False)

        if 'CPR_hash' not in df.columns:
            logger.warning("%s has no CPR_hash column — skipping", filename)
            continue

        # Ensure CPR_hash is string for consistent matching
        df['CPR_hash'] = df['CPR_hash'].astype(str)

        n_written = 0
        for cpr_hash, group in df.groupby('CPR_hash'):
            if cpr_hash not in cpr_hashes:
                continue

            out_dir = os.path.join(patient_dir, cpr_hash)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'{filename}.csv')
            group.to_csv(out_path, index=True)
            n_written += 1

        logger.info("  %s: wrote %d patient files", filename, n_written)

    logger.info("Split complete. Output: %s", patient_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Split population CSVs into per-patient directories'
    )
    parser.add_argument('--data-dir', default='data/raw',
                        help='Source directory with population CSVs')
    parser.add_argument('--patient-dir', default='data/patients',
                        help='Output directory for per-patient files')
    parser.add_argument('--subset', type=int, default=None,
                        help='Only split for the first N patients (for testing)')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--holdout', action='store_true',
                       help='Only split holdout patients (ServiceDate > split date)')
    group.add_argument('--trainval', action='store_true',
                       help='Only split trainval patients (ServiceDate <= split date)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    cpr_hashes = None

    if args.holdout or args.trainval:
        from astra.utils import get_cfg, get_base_df
        cfg = get_cfg()
        base_df = get_base_df()
        split_date = cfg.get('holdout_split_date', '2023-06-01')
        if args.holdout:
            mask = base_df['ServiceDate'] > split_date
            label = 'holdout'
        else:
            mask = base_df['ServiceDate'] <= split_date
            label = 'trainval'
        cpr_hashes = list(base_df.loc[mask, 'CPR_hash'].dropna().unique())
        logger.info("Splitting %s patients (%d unique CPR hashes, split_date=%s)",
                     label, len(cpr_hashes), split_date)

    if args.subset:
        if cpr_hashes is None:
            pi = pd.read_csv(
                os.path.join(args.data_dir, 'PatientInfo.csv'),
                usecols=['CPR_hash'], dtype={'CPR_hash': str},
            )
            cpr_hashes = list(pi['CPR_hash'].dropna().unique())
        cpr_hashes = cpr_hashes[:args.subset]
        logger.info("Subsetting to %d patients", len(cpr_hashes))

    split_population_csvs(
        data_dir=args.data_dir,
        patient_dir=args.patient_dir,
        cpr_hashes=cpr_hashes,
    )
