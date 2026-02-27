"""Validate that pre-hospital (PPJ) data is properly integrated into the ASTRA pipeline.

Checks intermediate files (pkl), base_df columns, bin_df timing, concept merges,
and final mapped outputs to ensure PPJ data flows through end-to-end.

Usage:
    python -m astra.data.validate_prehospital [--verbose]
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from astra.utils import cfg, get_base_df, setup_logging, ProjectManager

logger = logging.getLogger(__name__)


# ============================================================================
# Individual checks
# ============================================================================

def check_prehospital_pkl_files() -> dict:
    """Check that PPJ extraction produced non-empty intermediate files."""
    results = {}
    files = {
        "prehospital_VitaleVaerdier": "data/interim/prehospital_VitaleVaerdier.pkl",
        "prehospital_GCS": "data/interim/prehospital_GCS.pkl",
        "ppj_base_df": "data/interim/ppj_base_df.pkl",
    }

    for name, path in files.items():
        if not Path(path).exists():
            results[name] = {"status": "FAIL", "reason": f"File not found: {path}"}
            continue

        df = pd.read_pickle(path)
        n_rows = len(df)
        n_pids = df["PID"].nunique() if "PID" in df.columns else 0

        if n_rows == 0:
            results[name] = {"status": "FAIL", "reason": "Empty DataFrame (0 rows)"}
        else:
            results[name] = {
                "status": "PASS",
                "rows": n_rows,
                "patients": n_pids,
                "columns": df.columns.tolist(),
            }

    # Validate vitals content
    vitals_path = files["prehospital_VitaleVaerdier"]
    if Path(vitals_path).exists():
        vit = pd.read_pickle(vitals_path)
        if len(vit) > 0:
            expected_cols = {"TIMESTAMP", "PID", "FEATURE", "VALUE"}
            missing_cols = expected_cols - set(vit.columns)
            if missing_cols:
                results["prehospital_VitaleVaerdier"]["col_warning"] = f"Missing columns: {missing_cols}"

            features = set(vit["FEATURE"].unique())
            expected_features = {"SBP", "DBP", "HR", "SPO2"}
            found = features & expected_features
            missing = expected_features - features
            results["prehospital_VitaleVaerdier"]["features_found"] = sorted(found)
            if missing:
                results["prehospital_VitaleVaerdier"]["features_missing"] = sorted(missing)

            # Check timestamps are valid datetimes
            ts = pd.to_datetime(vit["TIMESTAMP"], errors="coerce")
            n_nat = ts.isna().sum()
            if n_nat > 0:
                results["prehospital_VitaleVaerdier"]["timestamp_warning"] = (
                    f"{n_nat}/{len(vit)} NaT timestamps"
                )

    # Validate GCS content
    gcs_path = files["prehospital_GCS"]
    if Path(gcs_path).exists():
        gcs = pd.read_pickle(gcs_path)
        if len(gcs) > 0:
            vals = pd.to_numeric(gcs["VALUE"], errors="coerce").dropna()
            if len(vals) > 0:
                results["prehospital_GCS"]["value_range"] = f"[{vals.min():.0f}, {vals.max():.0f}]"
                if vals.min() < 3 or vals.max() > 15:
                    results["prehospital_GCS"]["bounds_warning"] = "GCS values outside [3, 15]"

    # Validate ABCD content
    abcd_path = files["ppj_base_df"]
    if Path(abcd_path).exists():
        abcd = pd.read_pickle(abcd_path)
        if len(abcd) > 0:
            for col in ["A", "B", "C", "D"]:
                if col in abcd.columns:
                    n_valid = abcd[col].notna().sum()
                    unique = abcd[col].dropna().unique()[:10].tolist()
                    results["ppj_base_df"][f"{col}_valid"] = n_valid
                    results["ppj_base_df"][f"{col}_values"] = unique

    return results


def check_base_df() -> dict:
    """Verify base_df has prehospital columns with valid data."""
    results = {}

    base_path = cfg.get("base_df_path", "data/interim/base_df.pkl")
    if not Path(base_path).exists():
        return {"status": "FAIL", "reason": f"base_df not found: {base_path}"}

    base = pd.read_pickle(base_path)
    results["shape"] = base.shape

    # Check prehospital_start column
    if "prehospital_start" not in base.columns:
        results["prehospital_start"] = "FAIL — column missing"
    else:
        ph_start = base["prehospital_start"]
        n_valid = ph_start.notna().sum()
        n_before_admission = (ph_start < base["start"]).sum()
        results["prehospital_start"] = {
            "status": "PASS" if n_valid > 0 else "FAIL",
            "valid": f"{n_valid}/{len(base)}",
            "before_admission": n_before_admission,
        }

    # Check ABCD columns
    for col in ["A", "B", "C", "D"]:
        if col not in base.columns:
            results[f"ABCD_{col}"] = "FAIL — column missing"
        else:
            n_valid = base[col].notna().sum()
            n_na = (base[col] == "#na#").sum() if base[col].dtype == object else 0
            results[f"ABCD_{col}"] = {
                "valid": n_valid,
                "na_filled": n_na,
                "unique": base[col].dropna().unique()[:10].tolist(),
            }

    # Check prehospital_end
    if "prehospital_end" in base.columns:
        n_with_end = base["prehospital_end"].notna().sum()
        results["prehospital_end"] = f"{n_with_end}/{len(base)} patients"

    return results


def check_bin_df() -> dict:
    """Verify bin_df starts from prehospital_start for patients with PPJ data."""
    results = {}

    bin_path = "data/interim/mapped/bin_df.pkl"
    base_path = cfg.get("base_df_path", "data/interim/base_df.pkl")

    if not Path(bin_path).exists():
        return {"status": "FAIL", "reason": f"bin_df not found: {bin_path}"}
    if not Path(base_path).exists():
        return {"status": "FAIL", "reason": f"base_df not found: {base_path}"}

    bin_df = pd.read_pickle(bin_path)
    base = pd.read_pickle(base_path)

    results["bin_df_shape"] = bin_df.shape
    results["n_patients"] = bin_df["PID"].nunique()

    # For patients with prehospital_start < start, check that bins start earlier
    if "prehospital_start" in base.columns:
        ph_patients = base[base["prehospital_start"] < base["start"]].copy()
        n_ph = len(ph_patients)
        results["patients_with_earlier_start"] = n_ph

        if n_ph > 0:
            # Sample a few patients and verify their first bin starts near prehospital_start
            sample_pids = ph_patients["PID"].head(5).tolist()
            checks = []
            for pid in sample_pids:
                pid_bins = bin_df[bin_df["PID"] == pid].sort_values("bin_start")
                pid_base = base[base["PID"] == pid].iloc[0]
                if len(pid_bins) > 0:
                    first_bin = pid_bins.iloc[0]["bin_start"]
                    ph_start = pid_base["prehospital_start"]
                    hosp_start = pid_base["start"]
                    delta_hours = (hosp_start - first_bin).total_seconds() / 3600
                    checks.append({
                        "PID": pid,
                        "first_bin": str(first_bin),
                        "prehospital_start": str(ph_start),
                        "hospital_start": str(hosp_start),
                        "hours_before_admission": round(delta_hours, 1),
                    })
            results["sample_bin_checks"] = checks

    return results


def check_concept_files() -> dict:
    """Verify filtered concept files contain pre-hospital timestamps."""
    results = {}

    base_path = cfg.get("base_df_path", "data/interim/base_df.pkl")
    if not Path(base_path).exists():
        return {"status": "FAIL", "reason": "base_df not found"}
    base = pd.read_pickle(base_path)

    # Check VitaleVaerdier concept (should contain pre-hospital vitals)
    vit_path = "data/interim/concepts/VitaleVaerdier.pkl"
    if Path(vit_path).exists():
        vit = pd.read_pickle(vit_path)
        results["VitaleVaerdier"] = {"total_rows": len(vit)}

        if "prehospital_start" in base.columns and "TIMESTAMP" in vit.columns:
            # Find measurements that are before hospital admission
            merged = vit.merge(
                base[["PID", "start"]].drop_duplicates(),
                on="PID",
                how="left",
            )
            merged["TIMESTAMP"] = pd.to_datetime(merged["TIMESTAMP"], errors="coerce")
            pre_admission = merged[merged["TIMESTAMP"] < merged["start"]]
            results["VitaleVaerdier"]["pre_admission_rows"] = len(pre_admission)
            results["VitaleVaerdier"]["pre_admission_patients"] = (
                pre_admission["PID"].nunique() if len(pre_admission) > 0 else 0
            )

            if len(pre_admission) > 0:
                results["VitaleVaerdier"]["status"] = "PASS"
                features_before = pre_admission["FEATURE"].value_counts().to_dict()
                results["VitaleVaerdier"]["pre_admission_features"] = features_before
            else:
                results["VitaleVaerdier"]["status"] = "FAIL — no pre-admission vitals found"
    else:
        results["VitaleVaerdier"] = {"status": "FAIL", "reason": "File not found"}

    # Check ITAOversigtsrapport (should contain pre-hospital GCS)
    ita_path = "data/interim/concepts/ITAOversigtsrapport.pkl"
    if Path(ita_path).exists():
        ita = pd.read_pickle(ita_path)
        results["ITAOversigtsrapport"] = {"total_rows": len(ita)}

        if "prehospital_start" in base.columns and "TIMESTAMP" in ita.columns:
            merged = ita.merge(
                base[["PID", "start"]].drop_duplicates(),
                on="PID",
                how="left",
            )
            merged["TIMESTAMP"] = pd.to_datetime(merged["TIMESTAMP"], errors="coerce")
            pre_admission = merged[merged["TIMESTAMP"] < merged["start"]]
            results["ITAOversigtsrapport"]["pre_admission_rows"] = len(pre_admission)
            results["ITAOversigtsrapport"]["pre_admission_patients"] = (
                pre_admission["PID"].nunique() if len(pre_admission) > 0 else 0
            )

            if len(pre_admission) > 0:
                gcs_pre = pre_admission[pre_admission["FEATURE"] == "GCS"]
                results["ITAOversigtsrapport"]["pre_admission_GCS_rows"] = len(gcs_pre)
                results["ITAOversigtsrapport"]["status"] = "PASS"
            else:
                results["ITAOversigtsrapport"]["status"] = "FAIL — no pre-admission GCS found"
    else:
        results["ITAOversigtsrapport"] = {"status": "FAIL", "reason": "File not found"}

    return results


def check_mapped_outputs() -> dict:
    """Verify mapped output files have data in early bins (pre-hospital period)."""
    results = {}

    # Check VitaleVaerdier_mean (most common mapped output)
    for name in ["VitaleVaerdier_mean", "VitaleVaerdier_std"]:
        path = f"data/interim/mapped/{name}.pkl"
        if not Path(path).exists():
            results[name] = {"status": "SKIP", "reason": "File not found"}
            continue

        df = pd.read_pickle(path)
        results[name] = {"shape": df.shape}

        # Timestep columns are numeric (0, 1, 2, ...)
        ts_cols = [c for c in df.columns if isinstance(c, (int, float)) or str(c).isdigit()]
        if not ts_cols:
            ts_cols = [c for c in df.columns if c not in ("PID", "FEATURE")]
        results[name]["n_timesteps"] = len(ts_cols)

        if ts_cols:
            # Check first few bins for non-NaN data (pre-hospital period)
            early_bins = ts_cols[:6]  # First 6 bins (first hour at 10min resolution)
            early_data = df[early_bins]
            n_nonnan_early = early_data.notna().sum().sum()
            n_total_early = early_data.size
            results[name]["early_bins_fill_rate"] = f"{n_nonnan_early}/{n_total_early}"

            # Overall sparsity
            all_data = df[ts_cols]
            n_nonnan = all_data.notna().sum().sum()
            n_total = all_data.size
            results[name]["overall_fill_rate"] = f"{n_nonnan}/{n_total} ({100*n_nonnan/n_total:.1f}%)"

    return results


# ============================================================================
# Summary
# ============================================================================

def run_validation():
    """Run all prehospital validation checks and print summary."""
    print("=" * 70)
    print("  ASTRA Pre-Hospital Data Validation")
    print("=" * 70)
    print()

    all_pass = True

    # 1. PPJ intermediate files
    print("1. PPJ Intermediate Files (prehospital.py output)")
    print("-" * 50)
    pkl_results = check_prehospital_pkl_files()
    for name, info in pkl_results.items():
        if isinstance(info, dict):
            status = info.get("status", "?")
            icon = "OK" if status == "PASS" else "FAIL"
            if status != "PASS":
                all_pass = False
            print(f"  [{icon}] {name}")
            for k, v in info.items():
                if k != "status":
                    print(f"       {k}: {v}")
        else:
            print(f"  [??] {name}: {info}")
    print()

    # 2. base_df
    print("2. base_df Columns")
    print("-" * 50)
    base_results = check_base_df()
    for k, v in base_results.items():
        if isinstance(v, dict):
            status = v.get("status", "")
            icon = "OK" if "PASS" in str(status) else ("FAIL" if "FAIL" in str(status) else "  ")
            if "FAIL" in str(status):
                all_pass = False
            print(f"  [{icon}] {k}")
            for kk, vv in v.items():
                if kk != "status":
                    print(f"       {kk}: {vv}")
        elif "FAIL" in str(v):
            all_pass = False
            print(f"  [FAIL] {k}: {v}")
        else:
            print(f"  [  ] {k}: {v}")
    print()

    # 3. bin_df
    print("3. bin_df Timing")
    print("-" * 50)
    bin_results = check_bin_df()
    for k, v in bin_results.items():
        if k == "sample_bin_checks" and isinstance(v, list):
            print(f"  Sample patients (bins start before admission?):")
            for check in v:
                h = check["hours_before_admission"]
                icon = "OK" if h > 0 else "WARN"
                print(f"    [{icon}] PID {check['PID']}: first bin {h:.1f}h before admission")
        else:
            print(f"  {k}: {v}")
    print()

    # 4. Concept files
    print("4. Concept Files (after filter + prehospital merge)")
    print("-" * 50)
    concept_results = check_concept_files()
    for name, info in concept_results.items():
        if isinstance(info, dict):
            status = info.get("status", "?")
            icon = "OK" if "PASS" in str(status) else "FAIL"
            if "FAIL" in str(status):
                all_pass = False
            print(f"  [{icon}] {name}")
            for k, v in info.items():
                if k != "status":
                    print(f"       {k}: {v}")
        else:
            print(f"  [??] {name}: {info}")
    print()

    # 5. Mapped outputs
    print("5. Mapped Outputs (bin-aggregated)")
    print("-" * 50)
    mapped_results = check_mapped_outputs()
    for name, info in mapped_results.items():
        if isinstance(info, dict):
            status = info.get("status", "")
            if "SKIP" in str(status):
                print(f"  [SKIP] {name}: {info.get('reason', '')}")
            else:
                print(f"  [  ] {name}")
                for k, v in info.items():
                    if k != "status":
                        print(f"       {k}: {v}")
    print()

    # Final verdict
    print("=" * 70)
    if all_pass:
        print("  RESULT: ALL CHECKS PASSED")
        print("  Pre-hospital data is flowing through the pipeline.")
    else:
        print("  RESULT: SOME CHECKS FAILED")
        print("  Review the output above for details.")
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate pre-hospital data integration")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    pm = ProjectManager()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    success = run_validation()
    sys.exit(0 if success else 1)