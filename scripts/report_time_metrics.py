"""
Generate a natural-language description of time-dependent model performance.

Reads time_metrics CSVs produced by run_eval and summarises AUROC/AUPRC
ranges across configurable time intervals.

Usage:
    python scripts/report_time_metrics.py --model-name <name>
    python scripts/report_time_metrics.py --model-name <name> --active-only
    python scripts/report_time_metrics.py --model-name <name> --both
"""

import argparse
import sys
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


def main():
    parser = argparse.ArgumentParser(description="Report time-dependent metrics")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--active-only", action="store_true",
                        help="Report active-only cohort metrics")
    parser.add_argument("--both", action="store_true",
                        help="Report both full and active-only cohorts")
    parser.add_argument("--save", type=str, default=None,
                        help="Save output to file (default: stdout only)")
    args = parser.parse_args()

    output_lines = []

    if args.both or not args.active_only:
        df = _load_metrics(args.model_name, active_only=False)
        label = "full cohort" if args.both else ""
        text = generate_description(df, cohort_label=label)
        output_lines.append(text)

    if args.both or args.active_only:
        df_active = _load_metrics(args.model_name, active_only=True)
        label = "active patients only" if args.both else ""
        text = generate_description(df_active, cohort_label=label)
        output_lines.append(text)

    output = "\n\n".join(output_lines)
    print(output)

    if args.save:
        Path(args.save).write_text(output, encoding="utf-8")
        print(f"\nSaved to {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
