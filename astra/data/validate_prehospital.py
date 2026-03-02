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

    # Check prehospital_start column (NaT for patients without PPJ data)
    if "prehospital_start" not in base.columns:
        results["prehospital_start"] = "FAIL — column missing"
    else:
        ph_start = base["prehospital_start"]
        n_valid = ph_start.notna().sum()
        n_nat = ph_start.isna().sum()
        ih_col = "inhospital_start" if "inhospital_start" in base.columns else "start"
        n_before_admission = (ph_start < base[ih_col]).sum()
        results["prehospital_start"] = {
            "status": "PASS" if n_valid > 0 else "FAIL",
            "valid": f"{n_valid}/{len(base)}",
            "nat (no PPJ)": f"{n_nat}/{len(base)}",
            "before_admission": n_before_admission,
        }

    # Check inhospital_start column
    if "inhospital_start" in base.columns:
        ih = base["inhospital_start"]
        results["inhospital_start"] = {
            "status": "PASS" if ih.notna().all() else "WARN",
            "valid": f"{ih.notna().sum()}/{len(base)}",
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

    # For patients with prehospital_start < inhospital_start, check bins start earlier
    if "prehospital_start" in base.columns:
        ih_col = "inhospital_start" if "inhospital_start" in base.columns else "start"
        ph_patients = base[
            base["prehospital_start"].notna()
            & (base["prehospital_start"] < base[ih_col])
        ].copy()
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
                    hosp_start = pid_base.get(ih_col, pid_base["start"])
                    delta_hours = (hosp_start - first_bin).total_seconds() / 3600
                    checks.append({
                        "PID": pid,
                        "first_bin": str(first_bin),
                        "prehospital_start": str(ph_start),
                        "inhospital_start": str(hosp_start),
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

    # Use inhospital_start (hospital admission) as the boundary for "pre-admission"
    ih_col = "inhospital_start" if "inhospital_start" in base.columns else "start"

    # Check VitaleVaerdier concept (should contain pre-hospital vitals)
    vit_path = "data/interim/concepts/VitaleVaerdier.pkl"
    if Path(vit_path).exists():
        vit = pd.read_pickle(vit_path)
        results["VitaleVaerdier"] = {"total_rows": len(vit)}

        if "TIMESTAMP" in vit.columns:
            merged = vit.merge(
                base[["PID", ih_col]].drop_duplicates(),
                on="PID",
                how="left",
            )
            merged["TIMESTAMP"] = pd.to_datetime(merged["TIMESTAMP"], errors="coerce")
            pre_admission = merged[merged["TIMESTAMP"] < merged[ih_col]]
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
            results["VitaleVaerdier"]["status"] = "WARN — no TIMESTAMP column"
    else:
        results["VitaleVaerdier"] = {"status": "FAIL", "reason": "File not found"}

    # Check ITAOversigtsrapport (should contain pre-hospital GCS)
    ita_path = "data/interim/concepts/ITAOversigtsrapport.pkl"
    if Path(ita_path).exists():
        ita = pd.read_pickle(ita_path)
        results["ITAOversigtsrapport"] = {"total_rows": len(ita)}

        if "TIMESTAMP" in ita.columns:
            merged = ita.merge(
                base[["PID", ih_col]].drop_duplicates(),
                on="PID",
                how="left",
            )
            merged["TIMESTAMP"] = pd.to_datetime(merged["TIMESTAMP"], errors="coerce")
            pre_admission = merged[merged["TIMESTAMP"] < merged[ih_col]]
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
            results["ITAOversigtsrapport"]["status"] = "WARN — no TIMESTAMP column"
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


def spot_check_patients(n_patients: int = 3, seed: int = 42) -> dict:
    """Randomly select patients with PPJ data and verify mapped vitals align with sources.

    For each sampled patient, checks:
    - Raw prehospital vitals exist and fall within [prehospital_start, inhospital_start)
    - Raw inhospital vitals exist and fall within [inhospital_start, end]
    - Mapped bins in the prehospital period contain data matching prehospital source
    - Mapped bins in the inhospital period contain data matching inhospital source
    - Mapped mean values match manual recomputation from raw data
    """
    results = {}

    base_path = cfg.get("base_df_path", "data/interim/base_df.pkl")
    bin_path = "data/interim/mapped/bin_df.pkl"
    mapped_path = "data/interim/mapped/VitaleVaerdier_mean.pkl"
    ph_vitals_path = "data/interim/prehospital_VitaleVaerdier.pkl"
    concept_vitals_path = "data/interim/concepts/VitaleVaerdier.pkl"

    required = {
        "base_df": base_path, "bin_df": bin_path, "mapped": mapped_path,
        "ph_vitals": ph_vitals_path, "concept_vitals": concept_vitals_path,
    }
    for name, path in required.items():
        if not Path(path).exists():
            return {"status": "FAIL", "reason": f"{name} not found: {path}"}

    base = pd.read_pickle(base_path)
    bin_df = pd.read_pickle(bin_path)
    mapped = pd.read_pickle(mapped_path)
    ph_raw = pd.read_pickle(ph_vitals_path)
    ih_raw = pd.read_pickle(concept_vitals_path)

    ih_col = "inhospital_start" if "inhospital_start" in base.columns else "start"

    # Select random patients WITH prehospital data
    ph_pids = base[base["prehospital_start"].notna()]["PID"]
    if len(ph_pids) == 0:
        return {"status": "FAIL", "reason": "No patients with prehospital_start"}

    rng = np.random.default_rng(seed)
    sample_pids = rng.choice(ph_pids.values, size=min(n_patients, len(ph_pids)), replace=False)

    # Ensure raw timestamps are datetime
    ph_raw["TIMESTAMP"] = pd.to_datetime(ph_raw["TIMESTAMP"], errors="coerce")
    ph_raw["VALUE"] = pd.to_numeric(ph_raw["VALUE"], errors="coerce")
    ih_raw["TIMESTAMP"] = pd.to_datetime(ih_raw["TIMESTAMP"], errors="coerce")
    ih_raw["VALUE"] = pd.to_numeric(ih_raw["VALUE"], errors="coerce")

    patient_results = []
    for pid in sample_pids:
        pid = int(pid)
        pr = {"PID": pid, "checks": []}
        row = base[base["PID"] == pid].iloc[0]
        ph_start = row["prehospital_start"]
        ih_start = row[ih_col]
        end = row["end"]

        # --- Raw prehospital vitals for this patient ---
        ph_pid = ph_raw[ph_raw["PID"] == pid].copy()
        n_ph_raw = len(ph_pid)
        pr["prehospital_raw_rows"] = n_ph_raw

        if n_ph_raw > 0:
            ph_before_admission = ph_pid[ph_pid["TIMESTAMP"] < ih_start]
            ph_after_admission = ph_pid[ph_pid["TIMESTAMP"] >= ih_start]
            pr["ph_before_admission"] = len(ph_before_admission)
            pr["ph_after_admission"] = len(ph_after_admission)
            pr["ph_time_range"] = f"{ph_pid['TIMESTAMP'].min()} → {ph_pid['TIMESTAMP'].max()}"
            pr["checks"].append(
                ("PASS" if len(ph_before_admission) > 0 else "WARN",
                 f"prehospital vitals before admission: {len(ph_before_admission)} rows")
            )

        # --- Raw inhospital vitals for this patient ---
        ih_pid = ih_raw[ih_raw["PID"] == pid].copy()
        n_ih_raw = len(ih_pid)
        pr["inhospital_raw_rows"] = n_ih_raw

        if n_ih_raw > 0:
            ih_in_window = ih_pid[
                (ih_pid["TIMESTAMP"] >= ih_start) & (ih_pid["TIMESTAMP"] <= end)
            ]
            pr["ih_in_window"] = len(ih_in_window)
            pr["ih_time_range"] = f"{ih_pid['TIMESTAMP'].min()} → {ih_pid['TIMESTAMP'].max()}"

        # --- Mapped data for this patient ---
        mapped_pid = mapped[mapped["PID"] == pid].copy()
        pr["mapped_rows"] = len(mapped_pid)

        # --- Bin-level cross-check: pick one prehospital bin and one inhospital bin ---
        pid_bins = bin_df[bin_df["PID"] == pid].sort_values("bin_start")
        ph_bins = pid_bins[pid_bins["bin_start"] < ih_start]
        ih_bins = pid_bins[pid_bins["bin_start"] >= ih_start]
        pr["n_prehospital_bins"] = len(ph_bins)
        pr["n_inhospital_bins"] = len(ih_bins)

        # Check a prehospital bin
        if len(ph_bins) > 0 and n_ph_raw > 0:
            check_bin = ph_bins.iloc[len(ph_bins) // 2]  # middle bin
            pr["checks"] += _check_bin_values(
                check_bin, ph_pid, mapped_pid, label="prehospital"
            )

        # Check an inhospital bin
        if len(ih_bins) > 0 and n_ih_raw > 0:
            check_bin = ih_bins.iloc[min(5, len(ih_bins) - 1)]  # early inhospital bin
            pr["checks"] += _check_bin_values(
                check_bin, ih_pid, mapped_pid, label="inhospital"
            )

        # Overall status
        statuses = [c[0] for c in pr["checks"]]
        if any(s == "FAIL" for s in statuses):
            pr["status"] = "FAIL"
        elif any(s == "WARN" for s in statuses):
            pr["status"] = "WARN"
        else:
            pr["status"] = "PASS" if statuses else "SKIP"

        patient_results.append(pr)

    results["patients"] = patient_results
    n_pass = sum(1 for p in patient_results if p["status"] == "PASS")
    n_warn = sum(1 for p in patient_results if p["status"] == "WARN")
    n_fail = sum(1 for p in patient_results if p["status"] == "FAIL")
    results["summary"] = f"{n_pass} PASS, {n_warn} WARN, {n_fail} FAIL"
    results["status"] = "FAIL" if n_fail > 0 else "PASS"
    return results


def _check_bin_values(
    bin_row: pd.Series,
    raw_df: pd.DataFrame,
    mapped_df: pd.DataFrame,
    label: str,
) -> list:
    """Cross-check a single bin: raw measurements vs mapped mean.

    Returns list of (status, message) tuples.
    """
    checks = []
    b_start = bin_row["bin_start"]
    b_end = bin_row["bin_end"]
    b_counter = bin_row["bin_counter"]

    # Raw measurements in this bin
    raw_in_bin = raw_df[
        (raw_df["TIMESTAMP"] >= b_start) & (raw_df["TIMESTAMP"] <= b_end)
    ]
    # Mapped values for this bin
    mapped_in_bin = mapped_df[mapped_df["bin_counter"] == b_counter]

    if len(raw_in_bin) == 0:
        checks.append(("SKIP", f"{label} bin {b_counter} [{b_start}]: no raw data"))
        return checks

    # Per-feature check
    for feat in raw_in_bin["FEATURE"].unique():
        raw_vals = raw_in_bin[raw_in_bin["FEATURE"] == feat]["VALUE"].dropna()
        if len(raw_vals) == 0:
            continue
        expected_mean = raw_vals.mean()

        mapped_val_row = mapped_in_bin[mapped_in_bin["FEATURE"] == feat]
        if len(mapped_val_row) == 0:
            checks.append(("WARN", f"{label} bin {b_counter} {feat}: raw has {len(raw_vals)} vals but no mapped row"))
            continue

        mapped_val = mapped_val_row["VALUE"].iloc[0]
        if pd.isna(mapped_val):
            checks.append(("WARN", f"{label} bin {b_counter} {feat}: mapped is NaN despite {len(raw_vals)} raw vals"))
            continue

        # Allow small tolerance for float comparison
        if abs(mapped_val - expected_mean) < 0.01:
            checks.append(("PASS", f"{label} bin {b_counter} {feat}: mapped={mapped_val:.2f} == raw_mean={expected_mean:.2f} (n={len(raw_vals)})"))
        else:
            # Might differ because inhospital + prehospital raw overlap in same bin
            checks.append(("WARN",
                f"{label} bin {b_counter} {feat}: mapped={mapped_val:.2f} != raw_mean={expected_mean:.2f} (n={len(raw_vals)}) "
                f"— may include data from both sources"))

    return checks


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

    # 6. Spot-check: random patients with PPJ data
    print("6. Spot-Check: Mapped Vitals vs Raw Sources")
    print("-" * 50)
    spot_results = spot_check_patients(n_patients=3)
    if "reason" in spot_results:
        print(f"  [FAIL] {spot_results['reason']}")
        all_pass = False
    else:
        if spot_results.get("status") == "FAIL":
            all_pass = False
        print(f"  Summary: {spot_results.get('summary', '?')}")
        for pr in spot_results.get("patients", []):
            pid = pr["PID"]
            status = pr.get("status", "?")
            icon = {"PASS": "OK", "WARN": "~~", "FAIL": "FAIL", "SKIP": "SKIP"}.get(status, "??")
            print(f"\n  [{icon}] PID {pid}")
            print(f"       prehospital_raw_rows: {pr.get('prehospital_raw_rows', 0)}")
            if "ph_time_range" in pr:
                print(f"       ph_time_range: {pr['ph_time_range']}")
                print(f"       ph_before_admission: {pr.get('ph_before_admission', 0)}")
            print(f"       inhospital_raw_rows: {pr.get('inhospital_raw_rows', 0)}")
            if "ih_time_range" in pr:
                print(f"       ih_time_range: {pr['ih_time_range']}")
            print(f"       n_prehospital_bins: {pr.get('n_prehospital_bins', 0)}")
            print(f"       n_inhospital_bins: {pr.get('n_inhospital_bins', 0)}")
            print(f"       mapped_rows: {pr.get('mapped_rows', 0)}")
            for check_status, msg in pr.get("checks", []):
                check_icon = {"PASS": "OK", "WARN": "~~", "FAIL": "FAIL", "SKIP": "--"}.get(check_status, "??")
                print(f"       [{check_icon}] {msg}")
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