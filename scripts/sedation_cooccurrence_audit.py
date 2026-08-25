"""
Co-occurrence audit for sedation + surgical tier redesign.
Uses existing filtered concepts and bin_df — no pipeline re-run needed.

Run from project root:
    python scripts/sedation_cooccurrence_audit.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Load existing pipeline artefacts
# ---------------------------------------------------------------------------

concepts_path = Path("data/interim/concepts/Medicin.pkl")
bin_df_path = Path("data/interim/mapped/bin_df.pkl")
base_df_path = Path("data/interim/base_df.pkl")

med = pd.read_pickle(concepts_path)
bin_df = pd.read_pickle(bin_df_path)
base_df = pd.read_pickle(base_df_path)


print(f"Loaded {len(med):,} medication records, "
      f"{med['PID'].nunique():,} patients")
print(f"Columns: {list(med.columns)}")
print(f"ATC examples: {med['ATC'].dropna().head(10).tolist()}")

# Apply same filter as filter_medicin() — raw concepts file hasn't been filtered yet
from astra.data.mappings import MEDICATION_ACTION_LIST
med = med[med["Handling"].isin(MEDICATION_ACTION_LIST)].copy()
print(f"After action filter: {len(med):,} records")

# ---------------------------------------------------------------------------
# Assign medication records to bins via searchsorted (same as mapper.py)
# ---------------------------------------------------------------------------

# bin_df has per-patient bin boundaries
bin_df = bin_df.sort_values(["PID", "bin_counter"]).reset_index(drop=True)

# Build lookup: for each PID, bin_start array and bin_counter array
pid_bins = {}
for pid, grp in bin_df.groupby("PID"):
    starts = grp["bin_start"].values.astype("datetime64[ns]")
    counters = grp["bin_counter"].values
    pid_bins[pid] = (starts, counters)

# Filter med to patients in bin_df
valid_pids = set(pid_bins.keys())
med = med[med["PID"].isin(valid_pids)].copy()

# Raw concepts file uses Danish column names; filter_medicin() renames later
# Use Administrationstidspunkt as the timestamp
med["TIMESTAMP"] = pd.to_datetime(med["Administrationstidspunkt"], errors="coerce")
med = med.dropna(subset=["TIMESTAMP"])

# Assign bin_counter via searchsorted
def assign_bins(df, pid_bins):
    positions = []
    for _, row in df.iterrows():
        pid = row["PID"]
        ts = row["TIMESTAMP"]
        if pid not in pid_bins:
            positions.append(np.nan)
            continue
        starts, counters = pid_bins[pid]
        idx = np.searchsorted(starts, np.datetime64(ts), side="right") - 1
        if 0 <= idx < len(counters):
            positions.append(counters[idx])
        else:
            positions.append(np.nan)
    return positions

print("Assigning medication records to bins (this may take a minute)...")
# Vectorized approach: group by PID for efficiency
assigned = []
for pid, grp in med.groupby("PID"):
    if pid not in pid_bins:
        continue
    starts, counters = pid_bins[pid]
    ts_vals = grp["TIMESTAMP"].values.astype("datetime64[ns]")
    idxs = np.searchsorted(starts, ts_vals, side="right") - 1
    valid = (idxs >= 0) & (idxs < len(counters))
    sub = grp[valid].copy()
    sub["bin_counter"] = counters[idxs[valid]]
    assigned.append(sub)

df = pd.concat(assigned, ignore_index=True)
print(f"Assigned {len(df):,} records to bins")

# ---------------------------------------------------------------------------
# Reuse ATC code sets from composite_features.py
# ---------------------------------------------------------------------------

from astra.data.composite_features import (
    PROPOFOL, ESKETAMINE, NMBA_ALL, ETOMIDATE, THIOPENTAL, DEEP_BENZO,
    VOLATILE_PREFIX, ICU_OPIOID_PREFIX,
    WARD_SEDATION, ICU_LIGHT_SEDATION,
    PERIOP_VASOPRESSOR,
)

# Agent group definitions (matching proposed audit + our codebase)
VOLATILES = frozenset({c for c in df["ATC"].unique()
                       if isinstance(c, str) and c.startswith(VOLATILE_PREFIX)})
OR_OPIOIDS = frozenset({c for c in df["ATC"].unique()
                        if isinstance(c, str) and c.startswith(ICU_OPIOID_PREFIX)})

# Specific OR opioids for detailed breakdown
REMIFENTANIL = frozenset({"N01AH06"})
ALFENTANIL = frozenset({"N01AH02"})
SUFENTANIL = frozenset({"N01AH03"})
FENTANYL = frozenset({"N01AH01"})
# "True" OR opioids: agents used almost exclusively in GA context (excl fentanyl)
TRUE_OR_OPIOIDS = REMIFENTANIL | ALFENTANIL | SUFENTANIL
MIDAZOLAM = frozenset({"N05CD08"})
LORAZEPAM = frozenset({"N05BA06"})

print(f"\nDetected volatiles in cohort: {VOLATILES}")
print(f"Detected OR opioids in cohort: {OR_OPIOIDS}")

# ---------------------------------------------------------------------------
# Per-bin presence flags
# ---------------------------------------------------------------------------

def bin_flags(group):
    codes = set(group["ATC"].dropna())
    return pd.Series({
        "has_volatile":     bool(codes & VOLATILES),
        "has_propofol":     bool(codes & PROPOFOL),
        "has_esketamine":   bool(codes & ESKETAMINE),
        "has_nmba":         bool(codes & NMBA_ALL),
        "has_or_opioid":    bool(codes & OR_OPIOIDS),
        "has_fentanyl":     bool(codes & FENTANYL),
        "has_remifentanil": bool(codes & REMIFENTANIL),
        "has_true_or_opioid": bool(codes & TRUE_OR_OPIOIDS),
        "has_thiopental":   bool(codes & THIOPENTAL),
        "has_etomidate":    bool(codes & ETOMIDATE),
        "has_midazolam":    bool(codes & MIDAZOLAM),
        "has_lorazepam":    bool(codes & LORAZEPAM),
        "has_deep_benzo":   bool(codes & DEEP_BENZO),
        "has_periop_vaso":  bool(codes & PERIOP_VASOPRESSOR),
    })

print("\nComputing per-bin presence flags...")
bins = df.groupby(["PID", "bin_counter"]).apply(bin_flags).reset_index()
print(f"Total bins with any medication: {len(bins):,}")

# ---------------------------------------------------------------------------
# Co-occurrence pattern counts
# ---------------------------------------------------------------------------

patterns = {
    # Propofol patterns
    "propofol_alone":
        bins.has_propofol & ~bins.has_volatile & ~bins.has_nmba
        & ~bins.has_or_opioid & ~bins.has_esketamine,
    "propofol+volatile":
        bins.has_propofol & bins.has_volatile,
    "propofol+nmba":
        bins.has_propofol & bins.has_nmba,
    "propofol+or_opioid":
        bins.has_propofol & bins.has_or_opioid,
    "propofol+fentanyl_only":
        bins.has_propofol & bins.has_fentanyl
        & ~bins.has_or_opioid & ~bins.has_volatile & ~bins.has_nmba,
    "propofol+remifentanil":
        bins.has_propofol & bins.has_remifentanil,
    "propofol+volatile+nmba":
        bins.has_propofol & bins.has_volatile & bins.has_nmba,
    "propofol+or_opioid+nmba":
        bins.has_propofol & bins.has_or_opioid & bins.has_nmba,
    "propofol+or_opioid+volatile":
        bins.has_propofol & bins.has_or_opioid & bins.has_volatile,

    # Esketamine patterns
    "esketamine_alone":
        bins.has_esketamine & ~bins.has_propofol & ~bins.has_volatile
        & ~bins.has_nmba & ~bins.has_or_opioid,
    "esketamine+propofol":
        bins.has_esketamine & bins.has_propofol,
    "esketamine+nmba":
        bins.has_esketamine & bins.has_nmba,
    "esketamine+or_opioid":
        bins.has_esketamine & bins.has_or_opioid,

    # Other GA patterns
    "volatile_alone":
        bins.has_volatile & ~bins.has_propofol & ~bins.has_esketamine,
    "midazolam+nmba":
        bins.has_midazolam & bins.has_nmba,
    "thiopental_any":
        bins.has_thiopental,
    "etomidate_any":
        bins.has_etomidate,

    # Proposed new trigger: propofol + OR-opioid (TIVA pattern)
    "propofol+or_opioid_no_volatile":
        bins.has_propofol & bins.has_or_opioid & ~bins.has_volatile,

    # Fentanyl disambiguation
    "propofol+fentanyl_no_ga_marker":
        bins.has_propofol & bins.has_fentanyl
        & ~bins.has_true_or_opioid & ~bins.has_volatile
        & ~bins.has_nmba & ~bins.has_esketamine,
    "propofol+fentanyl+other_ga":
        bins.has_propofol & bins.has_fentanyl
        & (bins.has_true_or_opioid | bins.has_volatile | bins.has_nmba),
}

print("\n" + "=" * 70)
print("Bin-level co-occurrence patterns")
print("=" * 70)
print(f"{'Pattern':<45s} {'Bins':>8s} {'Patients':>10s}")
print("-" * 70)
for name, mask in patterns.items():
    n_bins = mask.sum()
    n_pts = bins.loc[mask, "PID"].nunique()
    print(f"  {name:<43s} {n_bins:>8,} {n_pts:>10,}")

# ---------------------------------------------------------------------------
# Current tier assignment (matching composite_features.py _derive_composite_tiers)
# ---------------------------------------------------------------------------

def assign_current_tiers(row):
    sed = 0
    surg = 0

    # Sedation: mirrors _derive_composite_tiers logic exactly
    if row.has_propofol and not row.has_volatile and not row.has_nmba:
        sed = max(sed, 3)
    if row.has_deep_benzo:
        sed = max(sed, 3)
    if row.has_esketamine and not row.has_volatile and not row.has_nmba and not row.has_propofol:
        sed = max(sed, 2)

    # Tier 4 triggers (current)
    tier4 = (
        row.has_volatile
        or row.has_thiopental
        or row.has_etomidate
        or (row.has_propofol and row.has_nmba)
        or (row.has_propofol and row.has_volatile)
        or (row.has_esketamine and row.has_nmba)
        or (row.has_esketamine and row.has_volatile)
        or (row.has_esketamine and row.has_propofol)
    )
    if tier4:
        sed = 4
    if tier4 and row.has_nmba:
        sed = 5

    # Surgical (current)
    if row.has_volatile or (row.has_esketamine and row.has_propofol):
        surg = max(surg, 2)
    if row.has_thiopental:
        surg = max(surg, 3)

    return pd.Series({"sed_current": sed, "surg_current": surg})


def assign_proposed_tiers(row):
    """Proposed: propofol + OR-opioid also triggers Tier 4 + Surgical 2."""
    sed = 0
    surg = 0

    # Same base tiers
    if row.has_propofol and not row.has_volatile and not row.has_nmba and not row.has_or_opioid:
        sed = max(sed, 3)
    if row.has_deep_benzo:
        sed = max(sed, 3)
    if row.has_esketamine and not row.has_volatile and not row.has_nmba and not row.has_propofol and not row.has_or_opioid:
        sed = max(sed, 2)

    # Tier 4 triggers (proposed: adds propofol+or_opioid)
    tier4 = (
        row.has_volatile
        or row.has_thiopental
        or row.has_etomidate
        or (row.has_propofol and row.has_nmba)
        or (row.has_propofol and row.has_volatile)
        or (row.has_propofol and row.has_or_opioid)  # PROPOSED
        or (row.has_esketamine and row.has_nmba)
        or (row.has_esketamine and row.has_volatile)
        or (row.has_esketamine and row.has_propofol)
    )
    if tier4:
        sed = 4
    if tier4 and row.has_nmba:
        sed = 5

    # Surgical (proposed: adds propofol+or_opioid)
    if row.has_volatile or (row.has_esketamine and row.has_propofol):
        surg = max(surg, 2)
    if row.has_propofol and row.has_or_opioid:  # PROPOSED
        surg = max(surg, 2)
    if row.has_thiopental:
        surg = max(surg, 3)

    return pd.Series({"sed_proposed": sed, "surg_proposed": surg})


print("\nAssigning current and proposed tiers...")
current = bins.apply(assign_current_tiers, axis=1)
proposed = bins.apply(assign_proposed_tiers, axis=1)
bins = pd.concat([bins, current, proposed], axis=1)

# ---------------------------------------------------------------------------
# Cross-tabs: sedation vs surgical independence
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("CURRENT: sedation_tier vs surgical_tier (bin counts)")
print("=" * 70)
print(pd.crosstab(bins.sed_current, bins.surg_current, margins=True))

print("\n" + "=" * 70)
print("PROPOSED: sedation_tier vs surgical_tier (bin counts)")
print("=" * 70)
print(pd.crosstab(bins.sed_proposed, bins.surg_proposed, margins=True))

# Patient-level (max tier per patient)
pt = bins.groupby("PID")[
    ["sed_current", "surg_current", "sed_proposed", "surg_proposed"]
].max()

print("\n" + "=" * 70)
print("CURRENT: sedation_tier vs surgical_tier (patient-level max)")
print("=" * 70)
print(pd.crosstab(pt.sed_current, pt.surg_current, margins=True))

print("\n" + "=" * 70)
print("PROPOSED: sedation_tier vs surgical_tier (patient-level max)")
print("=" * 70)
print(pd.crosstab(pt.sed_proposed, pt.surg_proposed, margins=True))

# ---------------------------------------------------------------------------
# Impact analysis: what changes between current and proposed?
# ---------------------------------------------------------------------------

# Bins that change sedation tier
sed_upgrade = bins["sed_proposed"] > bins["sed_current"]
sed_downgrade = bins["sed_proposed"] < bins["sed_current"]

print("\n" + "=" * 70)
print("TIVA IMPACT: Sedation tier changes")
print("=" * 70)
print(f"Bins upgraded:    {sed_upgrade.sum():>8,}  "
      f"({bins.loc[sed_upgrade, 'PID'].nunique():,} patients)")
print(f"Bins downgraded:  {sed_downgrade.sum():>8,}  "
      f"({bins.loc[sed_downgrade, 'PID'].nunique():,} patients)")
print(f"Bins unchanged:   {(~sed_upgrade & ~sed_downgrade).sum():>8,}")

# What tier transitions happen?
if sed_upgrade.any():
    transitions = bins.loc[sed_upgrade, ["sed_current", "sed_proposed"]].value_counts()
    print("\nUpgrade transitions:")
    for (old, new), count in transitions.items():
        n_pts = bins.loc[
            sed_upgrade & (bins.sed_current == old) & (bins.sed_proposed == new),
            "PID"
        ].nunique()
        print(f"  Tier {old} -> Tier {new}:  {count:>6,} bins  ({n_pts:,} patients)")

# Surgical tier changes
surg_upgrade = bins["surg_proposed"] > bins["surg_current"]
print(f"\nSurgical tier upgrades: {surg_upgrade.sum():>8,} bins  "
      f"({bins.loc[surg_upgrade, 'PID'].nunique():,} patients)")

# ---------------------------------------------------------------------------
# Propofol-alone vs propofol+OR-opioid breakdown
# ---------------------------------------------------------------------------

propofol_bins = bins[bins.has_propofol].copy()
n_propofol_bins = len(propofol_bins)
n_with_or_opioid = propofol_bins.has_or_opioid.sum()
n_with_volatile = propofol_bins.has_volatile.sum()
n_with_nmba = propofol_bins.has_nmba.sum()
n_truly_alone = (
    ~propofol_bins.has_or_opioid
    & ~propofol_bins.has_volatile
    & ~propofol_bins.has_nmba
    & ~propofol_bins.has_esketamine
).sum()

print("\n" + "=" * 70)
print("PROPOFOL CONTEXT BREAKDOWN")
print("=" * 70)
print(f"Total propofol bins:          {n_propofol_bins:>8,}")
print(f"  + OR-opioid:                {n_with_or_opioid:>8,}  "
      f"({100*n_with_or_opioid/max(n_propofol_bins,1):.1f}%)")
print(f"  + volatile:                 {n_with_volatile:>8,}  "
      f"({100*n_with_volatile/max(n_propofol_bins,1):.1f}%)")
print(f"  + NMBA:                     {n_with_nmba:>8,}  "
      f"({100*n_with_nmba/max(n_propofol_bins,1):.1f}%)")
print(f"  truly alone (no GA marker): {n_truly_alone:>8,}  "
      f"({100*n_truly_alone/max(n_propofol_bins,1):.1f}%)")

# OR-opioid detail within propofol bins
if n_with_or_opioid > 0:
    or_opioid_detail = propofol_bins[propofol_bins.has_or_opioid]
    print(f"\n  Within propofol+OR-opioid bins:")
    print(f"    also has volatile:  {or_opioid_detail.has_volatile.sum():>6,}")
    print(f"    also has NMBA:      {or_opioid_detail.has_nmba.sum():>6,}")
    pure_tiva = (
        or_opioid_detail.has_or_opioid
        & ~or_opioid_detail.has_volatile
        & ~or_opioid_detail.has_nmba
    ).sum()
    print(f"    pure TIVA (no vol/NMBA): {pure_tiva:>6,}  "
          f"<-- these flip T3->T4 under proposed rule")

# ---------------------------------------------------------------------------
# OR-opioid breakdown by specific agent
# ---------------------------------------------------------------------------

or_opioid_bins = df[df["ATC"].str.startswith("N01AH", na=False)].copy()
if len(or_opioid_bins) > 0:
    print("\n" + "=" * 70)
    print("OR-OPIOID ATC BREAKDOWN (record counts)")
    print("=" * 70)
    atc_counts = or_opioid_bins.groupby("ATC").agg(
        records=("PID", "size"),
        patients=("PID", "nunique"),
    ).sort_values("records", ascending=False)
    atc_labels = {
        "N01AH01": "fentanyl",
        "N01AH02": "alfentanil",
        "N01AH03": "sufentanil",
        "N01AH06": "remifentanil",
    }
    for atc, row in atc_counts.iterrows():
        label = atc_labels.get(atc, "unknown")
        print(f"  {atc} ({label:<14s}): {row.records:>8,} records, "
              f"{row.patients:>6,} patients")

# ---------------------------------------------------------------------------
# Fentanyl over-trigger analysis
# Propofol + fentanyl in ICU analgosedation (no other GA marker) would
# be incorrectly elevated to Tier 4 if fentanyl is included in OR-opioid rule
# ---------------------------------------------------------------------------

fent_no_ga = patterns["propofol+fentanyl_no_ga_marker"]
fent_with_ga = patterns["propofol+fentanyl+other_ga"]
n_fent_no_ga = fent_no_ga.sum()
n_fent_with_ga = fent_with_ga.sum()
n_fent_no_ga_pts = bins.loc[fent_no_ga, "PID"].nunique() if n_fent_no_ga > 0 else 0

print("\n" + "=" * 70)
print("FENTANYL OVER-TRIGGER ANALYSIS")
print("=" * 70)
print(f"Propofol + fentanyl ONLY (no other GA marker): "
      f"{n_fent_no_ga:>6,} bins, {n_fent_no_ga_pts:>5,} pts")
print(f"Propofol + fentanyl + other GA marker:         "
      f"{n_fent_with_ga:>6,} bins")
print(f"  --> {n_fent_no_ga:,} bins would over-trigger T3->T4 "
      f"if fentanyl is included in the OR-opioid rule")

# Compare: how many propofol+OR-opioid bins use ONLY true OR opioids vs fentanyl?
propofol_or = bins[bins.has_propofol & bins.has_or_opioid]
if len(propofol_or) > 0:
    n_with_true = propofol_or.has_true_or_opioid.sum()
    n_fent_only_in_or = (
        propofol_or.has_fentanyl
        & ~propofol_or.has_true_or_opioid
    ).sum()
    n_both = (propofol_or.has_fentanyl & propofol_or.has_true_or_opioid).sum()
    print(f"\nWithin propofol+OR-opioid bins ({len(propofol_or):,} total):")
    print(f"  has true OR opioid (remi/alfen/sufen): {n_with_true:>6,}")
    print(f"  has fentanyl only (no true OR opioid): {n_fent_only_in_or:>6,}  "
          f"<-- ambiguous (ICU or OR?)")
    print(f"  has both fentanyl + true OR opioid:    {n_both:>6,}")

# ---------------------------------------------------------------------------
# Mortality association (descriptive only)
# ---------------------------------------------------------------------------

target = "deceased_30d"
if target in base_df.columns:
    pt_outcome = base_df.set_index("PID")[target]
    pt["outcome"] = pt.index.map(pt_outcome)
    pt_valid = pt.dropna(subset=["outcome"])

    print("\n" + "=" * 70)
    print("MORTALITY BY MAX SEDATION TIER (current)")
    print("=" * 70)
    for tier in sorted(pt_valid.sed_current.unique()):
        sub = pt_valid[pt_valid.sed_current == tier]
        mort = sub.outcome.mean()
        print(f"  Tier {tier}: {len(sub):>6,} patients, "
              f"mortality {100*mort:.1f}%")

    print("\n" + "=" * 70)
    print("MORTALITY BY MAX SEDATION TIER (proposed)")
    print("=" * 70)
    for tier in sorted(pt_valid.sed_proposed.unique()):
        sub = pt_valid[pt_valid.sed_proposed == tier]
        mort = sub.outcome.mean()
        print(f"  Tier {tier}: {len(sub):>6,} patients, "
              f"mortality {100*mort:.1f}%")

    # Patients who change tier: mortality comparison
    changed = pt_valid[pt_valid.sed_proposed > pt_valid.sed_current]
    if len(changed) > 0:
        print(f"\nPatients whose max tier increases under proposed rule:")
        print(f"  N = {len(changed):,}, mortality = {100*changed.outcome.mean():.1f}%")
        unchanged = pt_valid[pt_valid.sed_proposed == pt_valid.sed_current]
        print(f"  (vs {len(unchanged):,} unchanged, "
              f"mortality = {100*unchanged.outcome.mean():.1f}%)")

# =============================================================================
# FINAL distribution check — fentanyl EXCLUDED from OR-opioid trigger
# Uses "anesthetic opioids" (remifentanil, alfentanil, sufentanil) only
# =============================================================================

print("\n\n")
print("#" * 70)
print("#  FINAL TIER SCHEME: fentanyl excluded from anesthetic opioid trigger")
print("#" * 70)

ANESTHETIC_OPIOIDS = TRUE_OR_OPIOIDS  # remi + alfen + sufen (defined above)

def final_bin_flags(group):
    codes = set(group["ATC"].dropna())
    return pd.Series({
        "has_volatile":      bool(codes & VOLATILES),
        "has_propofol":      bool(codes & PROPOFOL),
        "has_esketamine":    bool(codes & ESKETAMINE),
        "has_nmba":          bool(codes & NMBA_ALL),
        "has_anest_opioid":  bool(codes & ANESTHETIC_OPIOIDS),
        "has_fentanyl":      bool(codes & FENTANYL),
        "has_thiopental":    bool(codes & THIOPENTAL),
        "has_etomidate":     bool(codes & ETOMIDATE),
        "has_midazolam":     bool(codes & MIDAZOLAM),
        "has_lorazepam":     bool(codes & LORAZEPAM),
        "has_deep_benzo":    bool(codes & DEEP_BENZO),
        "has_dex":           bool(codes & frozenset({"N05CM18"})),
        "has_haloperidol":   bool(codes & frozenset({"N05AD01"})),
        "has_olanzapine":    bool(codes & frozenset({"N05AH03"})),
        "has_quetiapine":    bool(codes & frozenset({"N05AH04"})),
        "has_clonidine":     bool(codes & frozenset({"C02AC01"})),
        "has_ward_sed":      bool(codes & WARD_SEDATION),
    })

print("\nComputing final bin flags...")
fbins = df.groupby(["PID", "bin_counter"]).apply(final_bin_flags).reset_index()
print(f"Total bins with any medication: {len(fbins):,}")


def assign_final_tiers(row):
    sed = 0
    surg = 0

    # ---- Sedation tier ----
    # Fixed Tier 4
    if row.has_volatile:   sed = max(sed, 4)
    if row.has_thiopental: sed = max(sed, 4)
    if row.has_etomidate:  sed = max(sed, 4)

    # Co-occurrence Tier 4
    if row.has_propofol and row.has_volatile:      sed = max(sed, 4)
    if row.has_propofol and row.has_nmba:          sed = max(sed, 4)
    if row.has_propofol and row.has_anest_opioid:  sed = max(sed, 4)
    if row.has_esketamine and (
        row.has_nmba or row.has_volatile
        or row.has_propofol or row.has_anest_opioid
    ):
        sed = max(sed, 4)

    # Base tiers (only if not already elevated)
    if sed < 4:
        if row.has_propofol:    sed = max(sed, 3)
        if row.has_midazolam:   sed = max(sed, 3)
        if row.has_lorazepam:   sed = max(sed, 3)
    if sed < 3:
        if row.has_esketamine:  sed = max(sed, 2)
        if row.has_dex:         sed = max(sed, 2)
        if row.has_haloperidol: sed = max(sed, 2)
        if row.has_olanzapine:  sed = max(sed, 2)
        if row.has_quetiapine:  sed = max(sed, 2)
        if row.has_clonidine:   sed = max(sed, 2)
    if sed < 2:
        if row.has_ward_sed:    sed = max(sed, 1)

    # Tier 5: Tier 4 + NMBA
    if sed >= 4 and row.has_nmba:
        sed = 5

    # ---- Surgical tier ----
    if row.has_volatile:                             surg = max(surg, 2)
    if row.has_esketamine and row.has_propofol:       surg = max(surg, 2)
    if row.has_propofol and row.has_anest_opioid:     surg = max(surg, 2)
    if row.has_thiopental:                            surg = max(surg, 3)

    return pd.Series({"sed_tier": int(sed), "surg_tier": int(surg)})


print("Assigning final tiers...")
ftiers = fbins.apply(assign_final_tiers, axis=1)
fbins = pd.concat([fbins, ftiers], axis=1)

# --- Cross-tabs ---
print("\n" + "=" * 70)
print("FINAL: sedation_tier vs surgical_tier (bin counts)")
print("=" * 70)
print(pd.crosstab(fbins.sed_tier, fbins.surg_tier, margins=True))

fpt = fbins.groupby("PID")[["sed_tier", "surg_tier"]].max()

print("\n" + "=" * 70)
print("FINAL: sedation_tier vs surgical_tier (patient-level max)")
print("=" * 70)
print(pd.crosstab(fpt.sed_tier, fpt.surg_tier, margins=True))

# --- Per-tier distribution ---
print("\n" + "=" * 70)
print("FINAL: Sedation tier distribution")
print("=" * 70)
for t in sorted(fbins.sed_tier.unique()):
    n_bins = (fbins.sed_tier == t).sum()
    n_pts = fbins.loc[fbins.sed_tier == t, "PID"].nunique()
    print(f"  Tier {t}: {n_bins:>8,} bins, {n_pts:>6,} patients")

print("\n" + "=" * 70)
print("FINAL: Surgical tier distribution")
print("=" * 70)
for t in sorted(fbins.surg_tier.unique()):
    n_bins = (fbins.surg_tier == t).sum()
    n_pts = fbins.loc[fbins.surg_tier == t, "PID"].nunique()
    print(f"  Tier {t}: {n_bins:>8,} bins, {n_pts:>6,} patients")

# --- Mortality by tier ---
target = "deceased_30d"
if target in base_df.columns:
    fpt_outcome = base_df.set_index("PID")[target]
    fpt["outcome"] = fpt.index.map(fpt_outcome)
    fpt_valid = fpt.dropna(subset=["outcome"])

    print("\n" + "=" * 70)
    print("FINAL: Mortality by max sedation tier")
    print("=" * 70)
    for t in sorted(fpt_valid.sed_tier.unique()):
        sub = fpt_valid[fpt_valid.sed_tier == t]
        print(f"  Tier {t}: {len(sub):>6,} pts, "
              f"mortality {100*sub.outcome.mean():.1f}%")

    print("\n" + "=" * 70)
    print("FINAL: Mortality by max surgical tier")
    print("=" * 70)
    for t in sorted(fpt_valid.surg_tier.unique()):
        sub = fpt_valid[fpt_valid.surg_tier == t]
        print(f"  Tier {t}: {len(sub):>6,} pts, "
              f"mortality {100*sub.outcome.mean():.1f}%")

# --- Feature independence summary ---
print("\n" + "=" * 70)
print("INDEPENDENCE CHECK")
print("=" * 70)
surg_without_deep = ((fbins.surg_tier > 0) & (fbins.sed_tier < 4)).sum()
deep_without_surg = ((fbins.sed_tier >= 4) & (fbins.surg_tier == 0)).sum()
both = ((fbins.surg_tier > 0) & (fbins.sed_tier >= 4)).sum()
print(f"  Surgical > 0 AND sedation < 4:  {surg_without_deep:>6,} bins "
      f"(surgical without deep sed)")
print(f"  Sedation >= 4 AND surgical = 0: {deep_without_surg:>6,} bins "
      f"(deep sed without procedure)")
print(f"  Both sedation >= 4 AND surg > 0: {both:>6,} bins "
      f"(overlap)")

print("\nDone.")
