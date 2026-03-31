"""Count raw (pre-aggregation) registrations per concept for the model population.

Loads the cached data object to get trainval/holdout PIDs, then counts rows
in the filtered concept pickles (data/interim/concepts/) for those PIDs.

Usage:
    python -m scripts.count_registrations
"""

import pandas as pd
from pathlib import Path
from astra.data.caching import prepare_data_and_dls_cached
from astra.utils import get_cfg, setup_logging

CONCEPTS_DIR = Path("data/interim/concepts")

# Concepts of interest and their ID/datetime columns for context
CONCEPTS = {
    "Medicin": {"dt_col": "Administrationsstart"},
    "Labsvar": {"dt_col": "Prøvetagningstidspunkt"},
    "VitaleVaerdier": {"dt_col": "Registreringstidspunkt"},
    "EWS": {"dt_col": "Registreringstidspunkt"},
}


def load_concept(name: str) -> pd.DataFrame:
    pkl = CONCEPTS_DIR / f"{name}.pkl"
    if pkl.exists():
        return pd.read_pickle(pkl)
    csv = CONCEPTS_DIR / f"{name}.csv"
    if csv.exists():
        return pd.read_csv(csv, low_memory=False)
    raise FileNotFoundError(f"No file found for concept {name} in {CONCEPTS_DIR}")


def count_for_split(concept_df: pd.DataFrame, pids: set, label: str, concept: str):
    df = concept_df[concept_df["PID"].isin(pids)]
    n_rows = len(df)
    n_patients = df["PID"].nunique()

    # Try to find a feature column for breakdown
    feat_col = None
    for col in ["FEATURE", "Vital_parametre", "Analysenavn", "Generisk_navn"]:
        if col in df.columns:
            feat_col = col
            break

    print(f"  {label}: {n_rows:,} registrations across {n_patients:,} patients")

    if feat_col:
        top = df[feat_col].value_counts().head(10)
        print(f"    Top features ({feat_col}):")
        for feat, count in top.items():
            print(f"      {feat}: {count:,}")

    return n_rows, n_patients


def main():
    setup_logging("INFO")
    cfg = get_cfg()
    print("Loading cached data object...")
    data = prepare_data_and_dls_cached(cfg)

    trainval_pids = set(data["trainval"].base["PID"].unique())
    holdout_pids = set(data["holdout"].base["PID"].unique())
    all_pids = trainval_pids | holdout_pids

    print(f"\nPopulation: {len(trainval_pids):,} trainval + {len(holdout_pids):,} holdout = {len(all_pids):,} total patients\n")
    print("=" * 70)

    grand_total = 0
    for concept, meta in CONCEPTS.items():
        print(f"\n--- {concept} ---")
        try:
            df = load_concept(concept)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue

        total, _ = count_for_split(df, all_pids, "Total", concept)
        count_for_split(df, trainval_pids, "Trainval", concept)
        count_for_split(df, holdout_pids, "Holdout", concept)
        grand_total += total

    print(f"\n{'=' * 70}")
    print(f"GRAND TOTAL (Medicin + Labsvar + VitaleVaerdier + EWS): {grand_total:,} registrations")


if __name__ == "__main__":
    main()
