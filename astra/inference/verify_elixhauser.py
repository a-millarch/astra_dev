"""
Verify Python Elixhauser implementation against cached R-computed scores.

Loads the existing base_df.pkl (which contains R-computed ASMT_ELIX scores)
and recomputes Elixhauser scores using the new Python implementation for
every patient.  Prints a comparison table and summary statistics.

Run in Azure where data/interim/base_df.pkl and data/raw/Diagnoser.csv exist:

    python -m astra.inference.verify_elixhauser
"""

import numpy as np
import pandas as pd

from astra.inference.comorbidity import (
    compute_elixhauser_vw,
    map_elixhauser_categories,
)


def verify(base_df_path="data/interim/base_df.pkl", diag_path="data/raw/Diagnoser.csv"):
    # ---- Load cached R scores ----
    base = pd.read_pickle(base_df_path)
    print(f"Loaded base_df: {len(base)} patients")

    if "ASMT_ELIX" not in base.columns:
        print("ERROR: base_df has no ASMT_ELIX column — was Elixhauser run via R?")
        return

    r_scores = base.set_index("PID")["ASMT_ELIX"]

    # ---- Load raw diagnoses ----
    diag = pd.read_csv(diag_path)
    diag["Noteret_dato"] = pd.to_datetime(diag["Noteret_dato"], errors="coerce")
    diag["Løst_dato"] = pd.to_datetime(diag["Løst_dato"], errors="coerce")

    # ---- Recompute per patient using Python implementation ----
    results = []
    for _, row in base.iterrows():
        pid = row["PID"]
        cpr = row["CPR_hash"]
        start = pd.to_datetime(row["start"])

        # Replicate the same filtering as prepare_elix_df:
        # Diagnoses noted before trauma AND not resolved before trauma
        patient_diag = diag[diag["CPR_hash"] == cpr].copy()
        if patient_diag.empty:
            py_score = 0.0
            categories = set()
        else:
            filtered = patient_diag[
                (patient_diag["Noteret_dato"] <= start - pd.DateOffset(days=1))
                & (
                    patient_diag["Løst_dato"].isnull()
                    | (patient_diag["Løst_dato"] >= start + pd.DateOffset(days=1))
                )
            ]

            if filtered.empty:
                py_score = 0.0
                categories = set()
            else:
                # Strip first and last character (Danish ICD format → ICD-10)
                icd10_codes = (
                    filtered["Diagnosekode"]
                    .dropna()
                    .str.slice(1, -1)
                    .tolist()
                )
                py_score = compute_elixhauser_vw(icd10_codes)
                categories = map_elixhauser_categories(icd10_codes)

        r_score = r_scores.get(pid, np.nan)

        # R pipeline produces NaN when no diagnoses CSV existed at compute time,
        # while Python produces 0.0 (no matching diagnoses = score 0).
        # Treat NaN-vs-0 as a "compatible" match, not a real mismatch.
        r_is_nan = np.isnan(r_score) if isinstance(r_score, float) else pd.isna(r_score)
        if r_is_nan and py_score == 0.0:
            match_type = "nan_vs_zero"
        elif r_is_nan and np.isnan(py_score):
            match_type = "both_nan"
        elif not r_is_nan and r_score == py_score:
            match_type = "exact"
        else:
            match_type = "mismatch"

        results.append({
            "PID": pid,
            "R_score": r_score,
            "Python_score": py_score,
            "match_type": match_type,
            "diff": py_score - r_score if not r_is_nan else np.nan,
            "n_diagnoses": len(patient_diag),
            "categories": ", ".join(sorted(categories)) if categories else "",
        })

    df = pd.DataFrame(results)

    # ---- Summary ----
    total = len(df)
    exact = (df["match_type"] == "exact").sum()
    nan_vs_zero = (df["match_type"] == "nan_vs_zero").sum()
    both_nan = (df["match_type"] == "both_nan").sum()
    mismatched = (df["match_type"] == "mismatch").sum()
    nan_r = df["R_score"].isna().sum()

    print(f"\n{'='*60}")
    print(f"ELIXHAUSER VERIFICATION: Python vs R")
    print(f"{'='*60}")
    print(f"Total patients:      {total}")
    print(f"Exact matches:       {exact} ({100*exact/total:.1f}%)")
    print(f"NaN vs 0 (compat.):  {nan_vs_zero}")
    print(f"Both NaN:            {both_nan}")
    print(f"Real mismatches:     {mismatched}")
    print(f"R score is NaN:      {nan_r}")
    print(f"Effective accuracy:  {100*(exact + nan_vs_zero + both_nan)/total:.1f}%")

    if mismatched > 0:
        print(f"\n{'='*60}")
        print("MISMATCHED PATIENTS:")
        print(f"{'='*60}")
        mm = df[df["match_type"] == "mismatch"].sort_values("diff", key=abs, ascending=False)
        print(mm[["PID", "R_score", "Python_score", "diff", "n_diagnoses", "categories"]].to_string(index=False))

        print(f"\nDifference statistics:")
        diffs = mm["diff"].dropna()
        if len(diffs) > 0:
            print(f"  Mean abs diff:  {diffs.abs().mean():.2f}")
            print(f"  Max abs diff:   {diffs.abs().max():.2f}")
            print(f"  Median diff:    {diffs.median():.2f}")
    else:
        print("\nAll scores match (exact or NaN-vs-0 compatible)!")

    # Show a sample of matched scores for sanity
    print(f"\n{'='*60}")
    print("SAMPLE MATCHED SCORES (first 10 non-zero):")
    print(f"{'='*60}")
    nonzero = df[(df["match_type"] == "exact") & (df["Python_score"] != 0)].head(10)
    if len(nonzero) > 0:
        print(nonzero[["PID", "R_score", "Python_score", "categories"]].to_string(index=False))
    else:
        print("(no non-zero matched scores)")

    return df


if __name__ == "__main__":
    verify()