"""Categorical TS profile encoding.

Converts binary presence/absence categorical TS into ordinal profile levels
per category per time bin, based on clinician-defined rules.

Supports two profile types:

* **count / codes** (original) — ordinal levels from distinct sub-code counts
  or required sub-code combinations.  Output: int8 profile tensor.
* **tier_mapping** — maps individual ATC codes to clinically defined tiers
  using a registered ``TierMapping``, or uses auto-binary mode when no
  mapping is specified.  Output: continuous TS features (``max_level``,
  ``n_distinct``) injected as regular TS channels.

  tier_mapping config supports:
    mapping:      Name of a registered TierMapping (ordinal tiers).
                  If omitted, auto-binary: all matched records → tier 1.
    atc_prefixes: ATC prefix strings for scanning raw data. Supports
                  variable-length prefixes (e.g. "J01" or "C01CA03").
    atc_codes:    Shorthand for exact ATC codes (used as prefixes too).
                  Used when ``atc_prefixes`` is not specified.
    short_name:   Prefix for output feature names.
    features:     Per-bin aggregates: max_level, n_distinct.
    exclusions:   Code-level exclusion rules (e.g. dose-based).

Profile rules are defined in a separate YAML file (e.g., configs/profiles.yaml),
referenced from the main config via categorical_profiles.config_file.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Module-level cache for the profiles config
_profiles_config_cache: Optional[dict] = None


def load_profiles_config(cfg: dict) -> dict:
    """Load the profiles config from the YAML file referenced in cfg.

    Returns an empty dict if profiles are disabled or config_file is missing.
    The result is cached for the lifetime of the process.
    """
    global _profiles_config_cache

    if _profiles_config_cache is not None:
        return _profiles_config_cache

    profile_cfg = cfg.get("categorical_profiles", {})
    if not profile_cfg.get("enabled"):
        _profiles_config_cache = {}
        return _profiles_config_cache

    config_file = profile_cfg.get("config_file")
    if not config_file:
        logger.warning("categorical_profiles.enabled=true but no config_file specified")
        _profiles_config_cache = {}
        return _profiles_config_cache

    import yaml
    from astra.utils import PROJECT_ROOT

    config_path = PROJECT_ROOT / config_file
    if not config_path.exists():
        logger.warning(f"Profiles config not found: {config_path}")
        _profiles_config_cache = {}
        return _profiles_config_cache

    with open(config_path) as f:
        _profiles_config_cache = yaml.safe_load(f) or {}

    logger.info(f"Loaded profiles config from {config_path}")
    return _profiles_config_cache


def profiles_enabled(cfg: dict) -> bool:
    """Check if categorical profiles are globally enabled."""
    return cfg.get("categorical_profiles", {}).get("enabled", False)


def get_sub_code_level(cfg: dict, concept: str) -> int:
    """Get the sub_code_level for a concept, or 0 if profiles are disabled
    or the concept has no profile config.
    """
    if not profiles_enabled(cfg):
        return 0
    profiles = load_profiles_config(cfg)
    concept_cfg = profiles.get(concept, {})
    return concept_cfg.get("sub_code_level", 0)


def get_profiled_categories(cfg: dict, concept: str) -> Dict[str, list]:
    """Get the set of categories with profile rules for a concept.

    Returns:
        Dict mapping category name to its list of rule dicts.
        Empty dict if profiles disabled or no rules for this concept.
    """
    if not profiles_enabled(cfg):
        return {}
    profiles = load_profiles_config(cfg)
    concept_cfg = profiles.get(concept, {})
    return concept_cfg.get("categories", {})


def get_tier_feature_names(cfg: dict) -> Set[str]:
    """Collect all tier mapping feature channel names across all concepts.

    Used by dataloader.py to identify which channels should be excluded
    from trajectory length detection.

    When ``composite_mode: true`` is set for a concept, returns the 12
    composite feature names instead of per-category names.
    """
    names: Set[str] = set()
    if not profiles_enabled(cfg):
        return names
    profiles = load_profiles_config(cfg)
    for concept_name, concept_cfg in profiles.items():
        if not isinstance(concept_cfg, dict):
            continue
        if concept_cfg.get("composite_mode"):
            from astra.data.composite_features import COMPOSITE_FEATURE_NAMES
            names.update(COMPOSITE_FEATURE_NAMES)
            continue
        for _cat_name, cat_cfg in concept_cfg.get("categories", {}).items():
            tm = cat_cfg.get("tier_mapping")
            if not tm:
                continue
            short = tm.get("short_name", _cat_name)
            for feat in tm.get("features", []):
                names.add(f"{short}_{feat}")
    return names


def evaluate_profile_rules(rules: list, sub_codes: Set[str]) -> int:
    """Evaluate profile rules against a set of sub-codes and return the profile level.

    Rules are evaluated in descending level order; first match wins.

    Rule types:
        count: matches if min <= len(sub_codes) <= max
        codes: matches if all required codes are present in sub_codes

    Args:
        rules: List of rule dicts from the profiles config.
        sub_codes: Set of sub-code strings present in a bin for one category.

    Returns:
        The matching profile level (1-based), or 0 if no rule matches.
    """
    if not sub_codes:
        return 0

    count = len(sub_codes)

    # Sort rules by level descending so highest-priority match wins
    sorted_rules = sorted(rules, key=lambda r: r.get("level", 0), reverse=True)

    for rule in sorted_rules:
        rule_type = rule.get("type", "count")
        level = rule.get("level", 0)

        if rule_type == "count":
            rule_min = rule.get("min", 0)
            rule_max = rule.get("max", float("inf"))
            if rule_min <= count <= rule_max:
                return level

        elif rule_type == "codes":
            required = set(rule.get("required", []))
            if required and required.issubset(sub_codes):
                return level

    return 0


class CategoricalProfileEncoder:
    """Computes ordinal profile levels from sub-code detail per category per bin.

    This encoder sits between the wide-format pivot and MultiHotCategoricalEncoder.
    For each profiled category, it determines the profile level per (PID, timestep)
    based on the distinct sub-codes present.

    Non-profiled categories are passed through unchanged for binary multi-hot encoding.

    Supports two category types:

    * **count-based** (``rules`` key) — original ordinal profiling
    * **tier_mapping** (``tier_mapping`` key) — ATC code → tier classification
      producing continuous TS features
    """

    def __init__(self, concept_profile_config: dict):
        """
        Args:
            concept_profile_config: Profile config for one concept, e.g.::

                {
                    'sub_code_level': 4,
                    'categories': {
                        'antibiotics': {'tier_mapping': {...}},
                        'opiods': {'rules': [...]},
                    }
                }
        """
        self.config = concept_profile_config
        all_categories = concept_profile_config.get("categories", {})
        self.sub_code_level = concept_profile_config.get("sub_code_level", 0)

        # Separate count-based and tier_mapping categories
        self.count_categories: Dict[str, dict] = {}
        self.tier_categories: Dict[str, dict] = {}

        for cat_name, cat_cfg in all_categories.items():
            if "tier_mapping" in cat_cfg:
                self.tier_categories[cat_name] = cat_cfg["tier_mapping"]
            if "rules" in cat_cfg:
                self.count_categories[cat_name] = cat_cfg

        # For backward compat: profiled_categories includes ALL categories
        # that need stripping from the multi-hot encoding
        self.profiled_categories: Dict[str, dict] = all_categories

        # Compute max levels per count-based category (for tensor sizing)
        self.max_levels: Dict[str, int] = {}
        for cat_name, cat_cfg in self.count_categories.items():
            rules = cat_cfg.get("rules", [])
            if rules:
                self.max_levels[cat_name] = max(r.get("level", 0) for r in rules)

    def build_sub_code_index(
        self, long_df: pd.DataFrame
    ) -> Dict[Tuple, Dict[str, set]]:
        """Build an index mapping (PID, timestep) → {category: set(sub_codes)}
        from the long-format mapped data (BEFORE wide-format pivot).

        The long_df must have columns: PID, TIMESTEP, VALUE, and SUB_CODE.

        Returns:
            Dict[(pid, timestep)] → {category_name: {sub_code_1, sub_code_2, ...}}
        """
        if "SUB_CODE" not in long_df.columns:
            logger.warning("SUB_CODE column missing — cannot build sub-code index")
            return {}

        index: Dict[Tuple, Dict[str, set]] = {}

        for _, row in long_df.iterrows():
            key = (row["PID"], row["TIMESTEP"])
            category = row["VALUE"]
            sub_code = row.get("SUB_CODE")

            if pd.isna(category) or pd.isna(sub_code):
                continue

            if category not in self.count_categories:
                continue

            if key not in index:
                index[key] = {}
            if category not in index[key]:
                index[key][category] = set()
            index[key][category].add(sub_code)

        logger.info(
            f"Built sub-code index: {len(index)} (PID, timestep) entries "
            f"for {len(self.count_categories)} count-based profiled categories"
        )
        return index

    def build_sub_code_index_fast(
        self, long_df: pd.DataFrame
    ) -> Dict[Tuple, Dict[str, set]]:
        """Vectorized version of build_sub_code_index using groupby."""
        if "SUB_CODE" not in long_df.columns:
            logger.warning("SUB_CODE column missing — cannot build sub-code index")
            return {}

        # Filter to only count-based profiled categories
        profiled_mask = long_df["VALUE"].isin(self.count_categories)
        profiled_df = long_df[profiled_mask & long_df["SUB_CODE"].notna()].copy()

        if profiled_df.empty:
            return {}

        index: Dict[Tuple, Dict[str, set]] = {}

        grouped = profiled_df.groupby(["PID", "TIMESTEP", "VALUE"])["SUB_CODE"].apply(set)
        for (pid, ts, cat), sub_codes in grouped.items():
            key = (pid, ts)
            if key not in index:
                index[key] = {}
            index[key][cat] = sub_codes

        logger.info(
            f"Built sub-code index: {len(index)} (PID, timestep) entries "
            f"for {len(self.count_categories)} count-based profiled categories"
        )
        return index

    def compute_profiles(
        self,
        sub_code_index: Dict[Tuple, Dict[str, set]],
        pids: List,
        timestep_cols: List,
    ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        """Compute profile levels for all count-based profiled categories.

        Args:
            sub_code_index: Output of build_sub_code_index().
            pids: Sorted list of patient IDs.
            timestep_cols: List of timestep column identifiers.

        Returns:
            profile_array: np.ndarray [n_samples, n_profiled_categories, seq_len] (int8)
            profile_dims: Dict {category_name: n_levels} for model initialization
            category_order: List of profiled category names (order matches tensor dim 1)
        """
        category_order = sorted(self.max_levels.keys())
        n_samples = len(pids)
        n_profiled = len(category_order)
        seq_len = len(timestep_cols)

        profile_array = np.zeros((n_samples, n_profiled, seq_len), dtype=np.int8)

        pid_to_idx = {pid: idx for idx, pid in enumerate(pids)}

        for (pid, ts), cat_sub_codes in sub_code_index.items():
            if pid not in pid_to_idx:
                continue
            sample_idx = pid_to_idx[pid]
            ts_idx = ts if isinstance(ts, int) else timestep_cols.index(ts)

            if ts_idx < 0 or ts_idx >= seq_len:
                continue

            for cat_idx, cat_name in enumerate(category_order):
                if cat_name in cat_sub_codes:
                    rules = self.count_categories[cat_name].get("rules", [])
                    level = evaluate_profile_rules(rules, cat_sub_codes[cat_name])
                    profile_array[sample_idx, cat_idx, ts_idx] = level

        # Build profile_dims
        profile_dims = {cat: self.max_levels[cat] for cat in category_order}

        logger.info(
            f"Computed profiles: {n_samples} samples × {n_profiled} categories × "
            f"{seq_len} timesteps. Profile dims: {profile_dims}"
        )

        return profile_array, profile_dims, category_order

    # ------------------------------------------------------------------
    # Tier mapping features (continuous TS channels)
    # ------------------------------------------------------------------

    def compute_tier_features(
        self,
        concept_pkl_path: str,
        base_df: pd.DataFrame,
        cfg: dict,
    ) -> Optional[pd.DataFrame]:
        """Compute tier mapping features as continuous TS channels.

        Loads raw concept data (with full ATC codes and dose columns),
        classifies tiers, assigns to time bins, aggregates per bin.

        Args:
            concept_pkl_path: Path to the filtered concept pickle
                (e.g. ``data/interim/concepts/Medicin.pkl``).
            base_df: Base DataFrame for this split (trainval or holdout).
            cfg: Global config dict.

        Returns:
            Wide-format DataFrame ``[PID, FEATURE, target, '0', '1', ...]``
            with one row per PID per feature (e.g. ``abx_max_level``,
            ``abx_n_distinct``).  Returns ``None`` if no tier_mapping
            categories are configured or no data matches.
        """
        if not self.tier_categories:
            return None

        from astra.data.tier_mappings import get_mapping, validate_mapping
        from astra.data.mappings import ATC_LVL3_MAP, ATC_LVL4_MAP
        from astra.utils import get_bin_df

        target = cfg["target"]
        bin_freq_include = cfg.get("bin_freq_include", [])
        base_pids = set(base_df["PID"].unique())

        # Load raw concept data
        concept_df = pd.read_pickle(concept_pkl_path)
        concept_df = concept_df[concept_df["PID"].isin(base_pids)].copy()
        logger.info(
            f"Loaded {len(concept_df)} raw records from {concept_pkl_path} "
            f"for {concept_df['PID'].nunique()} patients"
        )

        if "ATC" not in concept_df.columns:
            logger.warning(
                f"ATC column missing in {concept_pkl_path} — "
                "cannot compute tier features"
            )
            return None

        # Ensure ATC is string
        concept_df["ATC"] = concept_df["ATC"].astype(str)

        # Load bin_df and filter to included frequencies
        bin_df = get_bin_df()
        bin_df = bin_df[bin_df["PID"].isin(base_pids)].copy()
        if bin_freq_include:
            bin_df = bin_df[bin_df["bin_freq"].isin(bin_freq_include)].copy()

        # Compute sequential positions (0-based) per patient
        bin_df = bin_df.sort_values(["PID", "bin_counter"])
        bin_df["position"] = bin_df.groupby("PID").cumcount()
        max_pos = bin_df["position"].max()
        ts_cols = [str(i) for i in range(max_pos + 1)]

        # Find timestamp column once (shared across all categories)
        ts_col = _find_timestamp_col(concept_df)

        # Assign bins ONCE for all records (avoid repeated per-category work)
        concept_df_binned = _assign_bins(concept_df, bin_df, ts_col)
        logger.info(
            f"Bin assignment: {len(concept_df_binned)}/{len(concept_df)} "
            f"records matched bins"
        )

        # Composite mode: delegate to composite_features module
        if self.config.get("composite_mode"):
            from astra.data.composite_features import compute_composite_features
            logger.info("Composite mode enabled — computing 12 composite features")
            return compute_composite_features(
                concept_df_binned, base_df, cfg, ts_cols, base_pids,
            )

        # Validate short_name uniqueness
        short_names = []
        for cat_name, tm_cfg in self.tier_categories.items():
            sn = tm_cfg.get("short_name", cat_name)
            if sn in short_names:
                logger.warning(
                    f"Duplicate short_name '{sn}' in tier categories — "
                    f"feature name collisions will occur"
                )
            short_names.append(sn)

        all_feature_rows = []
        _validated_mappings: set = set()

        for cat_name, tm_cfg in self.tier_categories.items():
            mapping_name = tm_cfg.get("mapping")
            short_name = tm_cfg.get("short_name", cat_name)
            features = tm_cfg.get("features", ["max_level"])
            exclusions = tm_cfg.get("exclusions", [])

            # Resolve ATC prefixes: config > atc_codes > ATC map fallback
            atc_prefixes = tm_cfg.get("atc_prefixes", [])
            if not atc_prefixes:
                atc_prefixes = tm_cfg.get("atc_codes", [])
            if not atc_prefixes:
                for source_map in [ATC_LVL3_MAP, ATC_LVL4_MAP]:
                    atc_prefixes.extend(source_map.get(cat_name, []))
            if not atc_prefixes:
                logger.warning(
                    f"No ATC prefixes for category '{cat_name}' — skipping"
                )
                continue

            # Filter pre-binned data to ATC codes matching this category
            prefix_mask = pd.Series(False, index=concept_df_binned.index)
            for pfx in atc_prefixes:
                prefix_mask |= concept_df_binned["ATC"].str.startswith(
                    pfx, na=False
                )
            cat_df = concept_df_binned[prefix_mask].copy()

            if cat_df.empty:
                logger.info(f"Category '{cat_name}': no records found")
                continue

            logger.info(
                f"Category '{cat_name}': {len(cat_df)} records, "
                f"{cat_df['PID'].nunique()} patients"
            )

            # Apply exclusions
            for excl in exclusions:
                cat_df = _apply_exclusion(cat_df, excl)

            # Deduplicate by (PID, timestamp, ATC)
            n_before = len(cat_df)
            cat_df = cat_df.drop_duplicates(
                subset=["PID", ts_col, "ATC"], keep="first"
            )
            if len(cat_df) < n_before:
                logger.info(
                    f"  Deduplicated: {n_before} → {len(cat_df)} records"
                )

            # Classify tiers: registered mapping or auto-binary
            if mapping_name:
                if mapping_name not in _validated_mappings:
                    validate_mapping(mapping_name)
                    _validated_mappings.add(mapping_name)
                mapping = get_mapping(mapping_name)

                cat_df["_tier"] = cat_df["ATC"].map(mapping.classify)
                n_total = len(cat_df)
                n_mapped = cat_df["_tier"].notna().sum()
                n_unmapped = n_total - n_mapped
                if n_unmapped > 0:
                    unmapped_codes = (
                        cat_df.loc[cat_df["_tier"].isna(), "ATC"]
                        .value_counts()
                        .head(10)
                    )
                    logger.warning(
                        f"  {n_unmapped}/{n_total} records unmapped "
                        f"({100 * n_unmapped / n_total:.2f}%). "
                        f"Top unmapped:\n{unmapped_codes}"
                    )
                cat_df = cat_df[cat_df["_tier"].notna()].copy()
                cat_df["_tier"] = cat_df["_tier"].astype(int)

                mapping_rate = n_mapped / n_total if n_total > 0 else 0
                logger.info(
                    f"  Mapping rate: {n_mapped}/{n_total} "
                    f"({100 * mapping_rate:.1f}%)"
                )
            else:
                # Auto-binary: all matched records are tier 1
                cat_df["_tier"] = 1

            if cat_df.empty:
                continue

            # Aggregate per (PID, position)
            feature_dfs = _aggregate_tier_features(
                cat_df, features, short_name, base_df, target, ts_cols, base_pids
            )
            all_feature_rows.extend(feature_dfs)

        if not all_feature_rows:
            return None

        result = pd.concat(all_feature_rows, ignore_index=True)
        result = result.sort_values(["PID", "FEATURE"]).reset_index(drop=True)

        n_features = result["FEATURE"].nunique()
        logger.info(
            f"Tier features: {len(result)} rows, "
            f"{result['PID'].nunique()} patients, {n_features} features, "
            f"{max_pos + 1} timesteps"
        )
        return result

    def strip_profiled_from_wide(
        self, df_wide: pd.DataFrame, timestep_cols: List
    ) -> pd.DataFrame:
        """Remove profiled category values from the wide DataFrame.

        Profiled categories are handled via the profile tensor or tier features,
        so they should not appear in the binary multi-hot encoding.  This function
        removes their values from the wide-format cells while keeping non-profiled
        values.

        When the concept config contains ``strip_categories``, those parent
        category names (matching ATC map keys like "cardiovascular_drugs") are
        used instead of the config category keys.  This is necessary because
        subcategory keys (e.g. "vasopressor_support") don't match the multi-hot
        values produced by ``filter_medicin()``.

        Args:
            df_wide: Wide DataFrame from _get_long_concept_df_multi_label()
            timestep_cols: Timestep column names

        Returns:
            Modified DataFrame with profiled category values removed from cells.
        """
        strip_cats = self.config.get("strip_categories")
        if strip_cats:
            profiled_set = set(strip_cats)
        elif self.profiled_categories:
            profiled_set = set(self.profiled_categories.keys())
        else:
            return df_wide
        df_out = df_wide.copy()

        for ts_col in timestep_cols:
            col_data = df_out[ts_col]
            new_col = []
            for cell in col_data:
                try:
                    is_na = bool(pd.isna(cell))
                except (ValueError, TypeError):
                    is_na = False
                if is_na:
                    new_col.append(cell)
                elif isinstance(cell, list):
                    filtered = [v for v in cell if v not in profiled_set]
                    new_col.append(filtered if filtered else np.nan)
                elif isinstance(cell, str) and cell in profiled_set:
                    new_col.append(np.nan)
                else:
                    new_col.append(cell)
            df_out[ts_col] = new_col

        return df_out


# ======================================================================
# Helper functions for tier mapping
# ======================================================================


def _find_timestamp_col(df: pd.DataFrame) -> str:
    """Find the timestamp column name in the raw concept DataFrame."""
    for col in ["Administrationstidspunkt", "TIMESTAMP", "start"]:
        if col in df.columns:
            return col
    raise ValueError(
        f"No recognized timestamp column found. "
        f"Available: {df.columns.tolist()}"
    )


def _apply_exclusion(df: pd.DataFrame, excl: dict) -> pd.DataFrame:
    """Apply a single exclusion rule to the concept DataFrame."""
    code = excl.get("code")
    condition = excl.get("condition")

    if not code or not condition:
        return df

    code_mask = df["ATC"].str.startswith(code, na=False)
    n_code = code_mask.sum()
    if n_code == 0:
        return df

    if condition == "dose_lte":
        threshold = excl.get("threshold")
        dose_col = excl.get("dose_column", "Administrationsdosis")
        unit_col = excl.get("unit_column")
        unit_value = excl.get("unit_value")

        if dose_col not in df.columns:
            logger.info(
                f"  Exclusion for {code}: dose column '{dose_col}' not found — "
                f"skipping exclusion (all {n_code} records kept)"
            )
            return df

        dose_mask = pd.to_numeric(df[dose_col], errors="coerce") <= threshold

        if unit_col and unit_value and unit_col in df.columns:
            unit_mask = df[unit_col].astype(str).str.strip() == unit_value
            exclude_mask = code_mask & dose_mask & unit_mask
        else:
            exclude_mask = code_mask & dose_mask

        n_excluded = exclude_mask.sum()
        logger.info(
            f"  Exclusion: {code} dose<={threshold}"
            + (f" ({unit_value})" if unit_value else "")
            + f" → {n_excluded}/{n_code} records excluded"
        )
        return df[~exclude_mask].copy()

    logger.warning(f"  Unknown exclusion condition: {condition}")
    return df


def _assign_bins(
    cat_df: pd.DataFrame,
    bin_df: pd.DataFrame,
    ts_col: str,
) -> pd.DataFrame:
    """Assign each record to a time bin using vectorized searchsorted.

    Replicates the bin assignment logic from mapper.py
    ``merge_and_aggregate_optimized``.
    """
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(cat_df[ts_col]):
        cat_df = cat_df.copy()
        cat_df[ts_col] = pd.to_datetime(cat_df[ts_col])

    cat_df = cat_df.copy()
    cat_df["PID"] = cat_df["PID"].astype("int32")
    bin_df = bin_df.copy()
    bin_df["PID"] = bin_df["PID"].astype("int32")

    bin_pids = set(bin_df["PID"].unique())
    cat_df = cat_df[cat_df["PID"].isin(bin_pids)]

    if cat_df.empty:
        return cat_df

    subset_grouped = cat_df.groupby("PID", sort=False)
    bin_grouped = bin_df.groupby("PID", sort=False)

    results = []
    for pid in cat_df["PID"].unique():
        try:
            patient_data = subset_grouped.get_group(pid)
            patient_bins = bin_grouped.get_group(pid)
        except KeyError:
            continue

        timestamps = patient_data[ts_col].values
        bin_starts = patient_bins["bin_start"].values
        bin_ends = patient_bins["bin_end"].values

        indices = np.searchsorted(bin_starts, timestamps, side="right") - 1
        valid_mask = (indices >= 0) & (indices < len(patient_bins))
        if not valid_mask.any():
            continue

        valid_indices = indices[valid_mask]
        within_bin = timestamps[valid_mask] < bin_ends[valid_indices]
        if not within_bin.any():
            continue

        final_mask = np.zeros(len(patient_data), dtype=bool)
        final_mask[np.where(valid_mask)[0][within_bin]] = True

        matched = patient_data[final_mask].copy()
        final_indices = valid_indices[within_bin]
        matched["_bin_position"] = patient_bins["position"].iloc[final_indices].values
        results.append(matched)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def _aggregate_tier_features(
    binned_df: pd.DataFrame,
    features: List[str],
    short_name: str,
    base_df: pd.DataFrame,
    target: str,
    ts_cols: List[str],
    base_pids: set,
) -> List[pd.DataFrame]:
    """Aggregate tier data per (PID, bin_position) into wide-format feature rows.

    Uses vectorized pivot_table instead of per-PID loops.

    Returns one DataFrame per requested feature, each with schema:
    ``[PID, FEATURE, target, '0', '1', '2', ...]``
    """
    result_dfs = []
    n_ts = len(ts_cols)
    sorted_pids = sorted(base_pids)
    target_map = base_df.set_index("PID")[target]

    for feat_name in features:
        feat_col = f"{short_name}_{feat_name}"

        if feat_name == "max_level":
            agg = binned_df.groupby(["PID", "_bin_position"])["_tier"].max()
        elif feat_name == "n_distinct":
            agg = binned_df.groupby(["PID", "_bin_position"])["ATC"].nunique()
        else:
            logger.warning(f"Unknown tier feature: {feat_name}")
            continue

        agg = agg.reset_index()
        agg.columns = ["PID", "position", "value"]

        # Vectorized pivot to wide format
        pivot = agg.pivot_table(
            index="PID", columns="position", values="value", aggfunc="first"
        )
        pivot = pivot.reindex(
            index=sorted_pids,
            columns=range(n_ts),
            fill_value=0.0,
        )
        pivot.columns = ts_cols

        # Build result DataFrame
        df = pivot.reset_index()
        df.rename(columns={"index": "PID"}, inplace=True)
        df["FEATURE"] = feat_col
        df[target] = df["PID"].map(target_map).astype(int)

        result_dfs.append(df)

        n_nonzero = (agg["value"] > 0).sum() if len(agg) > 0 else 0
        logger.info(
            f"  Feature '{feat_col}': "
            f"non-zero bins={n_nonzero}, "
            f"mean={agg['value'].mean():.2f}, max={agg['value'].max()}"
            if len(agg) > 0
            else f"  Feature '{feat_col}': no data"
        )

    return result_dfs
