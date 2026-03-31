"""
Generate a natural-language description of time-dependent model performance
and a supplementary table comparing full cohort vs active-only AUROC/AUPRC.

Reads time_metrics CSVs produced by run_eval.

Usage:
    python scripts/report_time_metrics.py --model-name <name>
    python scripts/report_time_metrics.py --model-name <name> --active-only
    python scripts/report_time_metrics.py --model-name <name> --both
    python scripts/report_time_metrics.py --model-name <name> --table
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# (label, start_hours, end_hours) — end is exclusive, None = unbounded
INTERVALS = [
    ("the first 12 hours",       0,    12),
    ("from 12 to 72 hours",     12,    72),
    ("from 72 hours to 7 days", 72,   168),
    ("from 7 days to 30 days", 168,   720),
    ("after 30 days",          720,  None),
]


def _load_metrics(model_name: str, active_only: bool) -> pd.DataFrame:
    suffix = "_active" if active_only else ""
    path = Path(f"reports/eval/{model_name}/predictions/time_metrics_{model_name}{suffix}.csv")
    if not path.exists():
        print(f"ERROR: {path} not found. Run evaluation first.", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def _describe_interval(df_slice: pd.DataFrame, label: str) -> str:
    auroc_lo = df_slice["auroc"].min()
    auroc_hi = df_slice["auroc"].max()
    auprc_lo = df_slice["auprc"].min()
    auprc_hi = df_slice["auprc"].max()
    return (
        f"{label} AUROC was {auroc_lo:.3f}-{auroc_hi:.3f} "
        f"and AUPRC {auprc_lo:.3f}-{auprc_hi:.3f}"
    )


def generate_description(df: pd.DataFrame, cohort_label: str = "") -> str:
    parts = []
    for label, start_h, end_h in INTERVALS:
        if end_h is not None:
            mask = (df["time_hours"] >= start_h) & (df["time_hours"] < end_h)
        else:
            mask = df["time_hours"] >= start_h
        subset = df[mask]
        if subset.empty:
            continue
        parts.append(_describe_interval(subset, label))

    if not parts:
        return "No data points found."

    # Stitch into flowing prose
    header = "The model performance varied depending on prediction time in the patient trajectory"
    if cohort_label:
        header += f" ({cohort_label})"
    body = ", ".join(parts[:-1])
    if len(parts) > 1:
        body += f", and stabilizing with {parts[-1].split('AUROC was ')[1]}"
        text = f"{header} with {body}."
    else:
        text = f"{header} with {parts[0]}."
    return text


# Key time points for the supplementary table (hours)
TABLE_TIMEPOINTS_H = [1, 3, 6, 12, 24, 48, 72, 168, 336, 720, 1440, 2160]
TABLE_TIMEPOINT_LABELS = {
    1: "1 h", 3: "3 h", 6: "6 h", 12: "12 h", 24: "24 h",
    48: "48 h", 72: "72 h", 168: "7 days", 336: "14 days",
    720: "30 days", 1440: "60 days", 2160: "90 days",
}


def _nearest_row(df: pd.DataFrame, target_hours: float) -> pd.Series | None:
    """Return the row closest to target_hours, or None if df is empty."""
    if df.empty:
        return None
    idx = (df["time_hours"] - target_hours).abs().idxmin()
    row = df.loc[idx]
    # Only accept if within half a step of the target
    if abs(row["time_hours"] - target_hours) > 1.5:
        return None
    return row


def _fmt_ci(val, ci_lo, ci_hi) -> str:
    return f"{val:.3f} [{ci_lo:.3f}\u2013{ci_hi:.3f}]"


def generate_supplementary_table(df_all: pd.DataFrame, df_active: pd.DataFrame) -> pd.DataFrame:
    """Build a side-by-side AUROC/AUPRC table at key time points."""
    rows = []
    for h in TABLE_TIMEPOINTS_H:
        label = TABLE_TIMEPOINT_LABELS[h]
        row_all = _nearest_row(df_all, h)
        row_act = _nearest_row(df_active, h)

        entry = {"Time": label}
        if row_all is not None:
            entry["AUROC (Full)"] = _fmt_ci(row_all["auroc"], row_all["auroc_ci_lower"], row_all["auroc_ci_upper"])
            entry["AUPRC (Full)"] = _fmt_ci(row_all["auprc"], row_all["auprc_ci_lower"], row_all["auprc_ci_upper"])
            entry["N (Full)"] = int(row_all["n_samples"])
        else:
            entry["AUROC (Full)"] = "—"
            entry["AUPRC (Full)"] = "—"
            entry["N (Full)"] = "—"

        if row_act is not None:
            entry["AUROC (Active)"] = _fmt_ci(row_act["auroc"], row_act["auroc_ci_lower"], row_act["auroc_ci_upper"])
            entry["AUPRC (Active)"] = _fmt_ci(row_act["auprc"], row_act["auprc_ci_lower"], row_act["auprc_ci_upper"])
            entry["N (Active)"] = int(row_act["n_samples"])
        else:
            entry["AUROC (Active)"] = "—"
            entry["AUPRC (Active)"] = "—"
            entry["N (Active)"] = "—"

        rows.append(entry)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Report time-dependent metrics")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--active-only", action="store_true",
                        help="Report active-only cohort metrics")
    parser.add_argument("--both", action="store_true",
                        help="Report both full and active-only cohorts")
    parser.add_argument("--table", action="store_true",
                        help="Generate supplementary table (full vs active-only)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save output to file (default: stdout only)")
    args = parser.parse_args()

    output_lines = []

    # ── Prose descriptions ──────────────────────────────────────────────
    if args.both or not (args.active_only or args.table):
        df = _load_metrics(args.model_name, active_only=False)
        label = "full cohort" if args.both else ""
        text = generate_description(df, cohort_label=label)
        output_lines.append(text)

    if args.both or args.active_only:
        df_active = _load_metrics(args.model_name, active_only=True)
        label = "active patients only" if args.both else ""
        text = generate_description(df_active, cohort_label=label)
        output_lines.append(text)

    # ── Supplementary table ─────────────────────────────────────────────
    if args.table:
        df_all = _load_metrics(args.model_name, active_only=False)
        df_act = _load_metrics(args.model_name, active_only=True)
        table = generate_supplementary_table(df_all, df_act)
        output_lines.append("\nSupplementary Table: Time-Dependent Predictive Performance\n")
        output_lines.append(table.to_string(index=False))

        # Also save as CSV
        csv_path = Path(f"reports/eval/{args.model_name}/supplementary_time_metrics.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path, index=False)
        print(f"Table CSV saved to {csv_path}", file=sys.stderr)

    output = "\n\n".join(output_lines)
    print(output)

    if args.save:
        Path(args.save).write_text(output, encoding="utf-8")
        print(f"\nSaved to {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
