"""
Quick analysis: how many vital sign values fall outside physiological ranges?

Loads the filtered VitaleVaerdier concept, applies filter_vitals,
then checks each feature against proposed bounds.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from astra.data.filters import filter_vitals
from astra.data.mappings import VITALS_MAP, HEIGHT_WEIGHT_MAP
from astra.utils import cfg, inches_to_cm, ounces_to_kg

# ── Proposed physiological bounds per feature ──────────────────────────
# Adjust these as you see fit before re-running
PROPOSED_BOUNDS = {
    "SBP":             (0, 300),
    "DBP":             (0, 200),
    "MAP":             (0, 300),
    "HR":              (0, 300),
    "SPO2":            (0, 100),
    "RESPIRATORYRATE": (0, 80),
    "TEMP":            (0, 45),
    "HEIGHT":          (50, 230),
    "WEIGHT":          (2, 300),
}

def main():
    # Load the in-hospital filtered concept
    vit = pd.read_pickle("data/interim/concepts/VitaleVaerdier.pkl")
    print(f"Loaded VitaleVaerdier concept: {len(vit):,} rows\n")

    # Check if EWS exists for augmentation
    ews = None
    try:
        ews = pd.read_pickle("data/interim/concepts/EWS.pkl")
        print(f"Loaded EWS concept: {len(ews):,} rows (will augment vitals)\n")
    except FileNotFoundError:
        print("No EWS file found, skipping EWS augmentation\n")

    # Extract HEIGHT/WEIGHT before filter_vitals drops them
    # (filter_vitals only keeps VITALS_MAP features, not HEIGHT_WEIGHT_MAP)
    raw_vit = vit.copy()
    raw_vit.rename(columns={"Værdi": "VALUE", "Vital_parametre": "FEATURE", "Registreringstidspunkt": "TIMESTAMP"}, inplace=True)
    raw_vit["FEATURE"] = raw_vit["FEATURE"].replace(to_replace=HEIGHT_WEIGHT_MAP)
    hw = raw_vit[raw_vit["FEATURE"].isin(HEIGHT_WEIGHT_MAP.values())][["TIMESTAMP", "PID", "FEATURE", "VALUE"]].copy()
    hw["VALUE"] = pd.to_numeric(hw["VALUE"], errors="coerce")
    hw.loc[hw.FEATURE == "HEIGHT", "VALUE"] = inches_to_cm(hw.loc[hw.FEATURE == "HEIGHT", "VALUE"])
    hw.loc[hw.FEATURE == "WEIGHT", "VALUE"] = ounces_to_kg(hw.loc[hw.FEATURE == "WEIGHT", "VALUE"])
    hw["VALUE"] = hw["VALUE"].astype(str)

    # Apply filter_vitals (same as make_data pipeline)
    vit = filter_vitals(vit, ews=ews)
    print(f"After filter_vitals: {len(vit):,} rows\n")

    # Append HEIGHT/WEIGHT
    vit = pd.concat([vit, hw], ignore_index=True)

    # Convert VALUE to numeric
    vit["VALUE_NUM"] = pd.to_numeric(vit["VALUE"], errors="coerce")

    # ── Per-feature analysis ──────────────────────────────────────────
    features = sorted(vit["FEATURE"].unique())

    print("=" * 90)
    print(f"{'FEATURE':<20} {'Count':>10} {'Min':>12} {'Max':>12} "
          f"{'Bounds':>16} {'Outside':>10} {'% Lost':>8}")
    print("=" * 90)

    total_before = 0
    total_outside = 0

    for feat in features:
        mask = vit["FEATURE"] == feat
        subset = vit.loc[mask, "VALUE_NUM"].dropna()
        n = len(subset)
        total_before += n

        vmin = subset.min() if n > 0 else np.nan
        vmax = subset.max() if n > 0 else np.nan

        if feat in PROPOSED_BOUNDS:
            lo, hi = PROPOSED_BOUNDS[feat]
            outside = ((subset < lo) | (subset > hi)).sum()
            total_outside += outside
            pct = 100.0 * outside / n if n > 0 else 0
            bounds_str = f"[{lo}, {hi}]"
            print(f"{feat:<20} {n:>10,} {vmin:>12.2f} {vmax:>12.2f} "
                  f"{bounds_str:>16} {outside:>10,} {pct:>7.3f}%")
        else:
            print(f"{feat:<20} {n:>10,} {vmin:>12.2f} {vmax:>12.2f} "
                  f"{'(no bounds)':>16} {'—':>10} {'—':>8}")

    print("=" * 90)
    pct_total = 100.0 * total_outside / total_before if total_before > 0 else 0
    print(f"{'TOTAL':<20} {total_before:>10,} {'':>12} {'':>12} "
          f"{'':>16} {total_outside:>10,} {pct_total:>7.3f}%")
    print()

    # ── Detailed breakdown: show distribution of out-of-range values ──
    print("\n── Detailed out-of-range breakdown ──\n")
    for feat in features:
        if feat not in PROPOSED_BOUNDS:
            continue

        lo, hi = PROPOSED_BOUNDS[feat]
        subset = vit.loc[vit["FEATURE"] == feat, "VALUE_NUM"].dropna()
        below = subset[subset < lo]
        above = subset[subset > hi]

        if len(below) == 0 and len(above) == 0:
            continue

        print(f"  {feat}  (bounds: [{lo}, {hi}])")
        if len(below) > 0:
            print(f"    Below {lo}: {len(below):,} values")
            print(f"      min={below.min():.2f}, max={below.max():.2f}, "
                  f"median={below.median():.2f}")
            # Show histogram-like buckets
            if len(below) > 1:
                buckets = pd.cut(below, bins=min(5, len(below.unique())), precision=1)
                print(f"      distribution: {buckets.value_counts().sort_index().to_dict()}")
        if len(above) > 0:
            print(f"    Above {hi}: {len(above):,} values")
            print(f"      min={above.min():.2f}, max={above.max():.2f}, "
                  f"median={above.median():.2f}")
            if len(above) > 1:
                buckets = pd.cut(above, bins=min(5, len(above.unique())), precision=1)
                print(f"      distribution: {buckets.value_counts().sort_index().to_dict()}")
        print()


if __name__ == "__main__":
    main()
