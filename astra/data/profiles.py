"""Categorical TS profile encoding.

Converts binary presence/absence categorical TS into ordinal profile levels
per category per time bin, based on clinician-defined rules.

Profile rules are defined in a separate YAML file (e.g., configs/profiles.yaml),
referenced from the main config via categorical_profiles.config_file.
"""

import logging
from pathlib import Path
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
    """

    def __init__(self, concept_profile_config: dict):
        """
        Args:
            concept_profile_config: Profile config for one concept, e.g.::

                {
                    'sub_code_level': 4,
                    'categories': {
                        'antibiotics': {'rules': [...]},
                        'opiods': {'rules': [...]},
                    }
                }
        """
        self.config = concept_profile_config
        self.profiled_categories: Dict[str, list] = concept_profile_config.get(
            "categories", {}
        )
        self.sub_code_level = concept_profile_config.get("sub_code_level", 0)

        # Compute max levels per profiled category (for tensor sizing)
        self.max_levels: Dict[str, int] = {}
        for cat_name, cat_cfg in self.profiled_categories.items():
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

            if category not in self.profiled_categories:
                continue

            if key not in index:
                index[key] = {}
            if category not in index[key]:
                index[key][category] = set()
            index[key][category].add(sub_code)

        logger.info(
            f"Built sub-code index: {len(index)} (PID, timestep) entries "
            f"for {len(self.profiled_categories)} profiled categories"
        )
        return index

    def build_sub_code_index_fast(
        self, long_df: pd.DataFrame
    ) -> Dict[Tuple, Dict[str, set]]:
        """Vectorized version of build_sub_code_index using groupby."""
        if "SUB_CODE" not in long_df.columns:
            logger.warning("SUB_CODE column missing — cannot build sub-code index")
            return {}

        # Filter to only profiled categories
        profiled_mask = long_df["VALUE"].isin(self.profiled_categories)
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
            f"for {len(self.profiled_categories)} profiled categories"
        )
        return index

    def compute_profiles(
        self,
        sub_code_index: Dict[Tuple, Dict[str, set]],
        pids: List,
        timestep_cols: List,
    ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        """Compute profile levels for all profiled categories.

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
                    rules = self.profiled_categories[cat_name].get("rules", [])
                    level = evaluate_profile_rules(rules, cat_sub_codes[cat_name])
                    profile_array[sample_idx, cat_idx, ts_idx] = level

        # Build profile_dims
        profile_dims = {cat: self.max_levels[cat] for cat in category_order}

        logger.info(
            f"Computed profiles: {n_samples} samples × {n_profiled} categories × "
            f"{seq_len} timesteps. Profile dims: {profile_dims}"
        )

        return profile_array, profile_dims, category_order

    def strip_profiled_from_wide(
        self, df_wide: pd.DataFrame, timestep_cols: List
    ) -> pd.DataFrame:
        """Remove profiled category values from the wide DataFrame.

        Profiled categories are handled via the profile tensor, so they should
        not appear in the binary multi-hot encoding. This function removes their
        values from the wide-format cells while keeping non-profiled values.

        Args:
            df_wide: Wide DataFrame from _get_long_concept_df_multi_label()
            timestep_cols: Timestep column names

        Returns:
            Modified DataFrame with profiled category values removed from cells.
        """
        if not self.profiled_categories:
            return df_wide

        profiled_set = set(self.profiled_categories.keys())
        df_out = df_wide.copy()

        for ts_col in timestep_cols:
            col_data = df_out[ts_col]
            new_col = []
            for cell in col_data:
                if pd.isna(cell):
                    new_col.append(cell)
                elif isinstance(cell, list):
                    filtered = [v for v in cell if v not in profiled_set]
                    new_col.append(filtered if filtered else np.nan)
                elif cell in profiled_set:
                    new_col.append(np.nan)
                else:
                    new_col.append(cell)
            df_out[ts_col] = new_col

        return df_out
