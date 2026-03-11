"""
Diagnostic: compare EBM feature DataFrames between batch and inference paths
at a specific masking interval for a single holdout patient.

Reveals whether the EBM prediction divergence is caused by feature differences,
model differences (K-fold averaged vs deployment), or both.

Usage (in notebook after loading data + session + ctx):
    from scripts.diagnose_ebm_features import compare_ebm_features
    compare_ebm_features(data, session, ctx, cpr_hash, masking_hours=2.0)
"""

import os
import pickle

import numpy as np
import pandas as pd

from astra.utils import get_cfg


def compare_ebm_features(data, session, ctx, cpr_hash, masking_hours=2.0):
    """Compare batch vs inference EBM features at a specific interval.

    Args:
        data: Batch data dict from prepare_data_and_dls_cached().
        session: InferenceSession with loaded model.
        ctx: PatientContext built from CSV.
        cpr_hash: Full CPR hash string.
        masking_hours: EBM masking interval to compare (default 2.0h).
    """
    cfg = get_cfg()

    # ---- Find patient ----
    batch_base = pd.read_pickle("data/interim/base_df.pkl")
    matches = batch_base[batch_base["CPR_hash"] == cpr_hash]
    if matches.empty:
        matches = batch_base[batch_base["CPR_hash"].str.startswith(cpr_hash[:16])]
    if matches.empty:
        print(f"ERROR: CPR_hash {cpr_hash[:8]}... not found")
        return

    batch_pid = int(matches.iloc[0]["PID"])
    print(f"Patient: CPR={cpr_hash[:10]}... batch_PID={batch_pid}")
    print(f"Comparing EBM features at masking_hours={masking_hours}h")

    # ---- Batch path: extract features via AggregatedDS ----
    from astra.models.ebm.generate_ebm_feature import (
        _create_aggregated_dataset,
        _pad_to_reference_features,
        preprocess_features,
        generate_ebm_intervals,
        _model_filename,
    )
    from astra.utils import get_train_test_split

    base_df_full = data.get("base_df", batch_base)
    trainval_df, holdout_df = get_train_test_split(cfg, base_df_full)

    # Reference features from latest interval
    intervals = generate_ebm_intervals(cfg)
    _, _, ref_cat_feats, ref_cont_feats = _create_aggregated_dataset(
        trainval_df, cfg, intervals[-1]
    )

    # Single-patient holdout features
    patient_holdout = holdout_df[holdout_df["PID"] == batch_pid]
    if patient_holdout.empty:
        print(f"ERROR: PID {batch_pid} not in holdout")
        return

    X_batch_full, _, cat_feats_b, cont_feats_b = _create_aggregated_dataset(
        patient_holdout, cfg, masking_hours
    )
    id_col = cfg["dataset"]["id_col"]
    X_batch = X_batch_full.drop(columns=[id_col])
    X_batch, _, _ = _pad_to_reference_features(
        X_batch, cat_feats_b, cont_feats_b, ref_cat_feats, ref_cont_feats
    )

    # ---- Inference path: extract features via _aggregate_patient_features ----
    from astra.inference.ebm import _aggregate_patient_features
    from astra.data.datasets import get_effective_cat_cols

    filtered_concepts = ctx._ebm_context["filtered_concepts"]
    inf_base_df = ctx._ebm_context["base_df"]
    admission_time = pd.Timestamp(ctx.raw_data["admission_time"])
    ts_cat_names = set(cfg.get("dataset", {}).get("ts_cat_names", []))

    X_inf_full = _aggregate_patient_features(
        filtered_concepts=filtered_concepts,
        base_df=inf_base_df,
        admission_time=admission_time,
        masking_hours=masking_hours,
        ts_cat_names=ts_cat_names,
        cfg=cfg,
    )
    X_inf = X_inf_full.drop(columns=[id_col], errors="ignore")

    # ---- Compare raw features (before preprocessing) ----
    _section("RAW FEATURE COMPARISON (before one-hot encoding)")

    batch_cols = set(X_batch.columns)
    inf_cols = set(X_inf.columns)
    only_batch = sorted(batch_cols - inf_cols)
    only_inf = sorted(inf_cols - batch_cols)
    common = sorted(batch_cols & inf_cols)

    print(f"Batch columns: {len(batch_cols)}")
    print(f"Inference columns: {len(inf_cols)}")
    print(f"Common: {len(common)}")

    if only_batch:
        print(f"\nColumns ONLY in batch ({len(only_batch)}):")
        for c in only_batch[:30]:
            val = X_batch[c].iloc[0]
            print(f"  {c}: {val}")

    if only_inf:
        print(f"\nColumns ONLY in inference ({len(only_inf)}):")
        for c in only_inf[:30]:
            val = X_inf[c].iloc[0]
            print(f"  {c}: {val}")

    # Value differences in common columns
    diffs = []
    for col in common:
        bv = X_batch[col].iloc[0]
        iv = X_inf[col].iloc[0]
        try:
            bv_f, iv_f = float(bv), float(iv)
            delta = abs(bv_f - iv_f)
        except (ValueError, TypeError):
            delta = 0.0 if str(bv) == str(iv) else float("inf")
            bv_f, iv_f = bv, iv
        diffs.append({"column": col, "batch": bv_f, "inference": iv_f, "abs_diff": delta})

    diff_df = pd.DataFrame(diffs).sort_values("abs_diff", ascending=False)
    nonzero_diffs = diff_df[diff_df["abs_diff"] > 1e-6]

    if len(nonzero_diffs) > 0:
        print(f"\nFeatures with value differences ({len(nonzero_diffs)}):")
        print(nonzero_diffs.head(30).to_string(index=False))
    else:
        print("\nAll common features match perfectly!")

    # ---- Deployment model predictions on both feature sets ----
    _section("DEPLOYMENT MODEL PREDICTIONS")

    ebm_models_dir = cfg.get("ebm_feature", {}).get("models_dir", "models/ebm")
    model_path = os.path.join(ebm_models_dir, _model_filename(masking_hours))
    if not os.path.exists(model_path):
        print(f"Deployment model not found: {model_path}")
        return

    with open(model_path, "rb") as f:
        model_dict = pickle.load(f)

    # Process batch features through deployment model
    our_cat_b = [c for c in X_batch.columns if c in model_dict["expected_cat_feats"]]
    our_cont_b = [c for c in X_batch.columns if c in model_dict["expected_cont_feats"]]
    X_proc_b, _, _ = preprocess_features(
        X_batch, our_cat_b, our_cont_b,
        encoder=model_dict["encoder"], fit=False,
        expected_cat_feats=model_dict["expected_cat_feats"],
        expected_cont_feats=model_dict["expected_cont_feats"],
    )

    # Process inference features through deployment model
    our_cat_i = [c for c in X_inf.columns if c in model_dict["expected_cat_feats"]]
    our_cont_i = [c for c in X_inf.columns if c in model_dict["expected_cont_feats"]]
    X_proc_i, _, _ = preprocess_features(
        X_inf, our_cat_i, our_cont_i,
        encoder=model_dict["encoder"], fit=False,
        expected_cat_feats=model_dict["expected_cat_feats"],
        expected_cont_feats=model_dict["expected_cont_feats"],
    )

    prob_b = float(model_dict["model"].predict_proba(X_proc_b)[:, 1][0])
    prob_i = float(model_dict["model"].predict_proba(X_proc_i)[:, 1][0])

    print(f"Deployment model on BATCH features:     {prob_b:.6f}")
    print(f"Deployment model on INFERENCE features:  {prob_i:.6f}")
    print(f"Difference:                              {abs(prob_b - prob_i):.6f}")

    # Compare with stored predictions
    ebm_save_dir = cfg.get("ebm_feature", {}).get("save_dir", "data/interim/ebm_features")
    preds_path = os.path.join(ebm_save_dir, "ebm_predictions.pkl")
    if os.path.exists(preds_path):
        with open(preds_path, "rb") as f:
            stored = pickle.load(f)
        stored_pred = stored.get("holdout", {}).get(batch_pid, {}).get(masking_hours)
        inf_pred = ctx._ebm_cache.get(masking_hours) if hasattr(ctx, "_ebm_cache") and ctx._ebm_cache else None
        print(f"\nStored batch prediction (ebm_predictions.pkl): {stored_pred}")
        print(f"Inference prediction (ctx._ebm_cache):          {inf_pred}")
        if stored_pred is not None:
            print(f"Stored vs deployment-on-batch-feats diff:       {abs(stored_pred - prob_b):.6f}")
            print("  (If large: stored preds used K-fold models, not deployment)")

    # ---- Processed feature comparison ----
    _section("PROCESSED FEATURE COMPARISON (after one-hot encoding)")
    proc_diff = (X_proc_b.values.astype(float) - X_proc_i.values.astype(float))
    max_diff = np.abs(proc_diff).max()
    n_diff = int((np.abs(proc_diff) > 1e-6).sum())
    print(f"Max processed feature difference: {max_diff:.6f}")
    print(f"Positions with diff > 1e-6: {n_diff} / {proc_diff.size}")

    if n_diff > 0:
        feat_names = list(X_proc_b.columns)
        for j in range(proc_diff.shape[1]):
            d = abs(proc_diff[0, j])
            if d > 1e-6:
                print(f"  {feat_names[j]}: batch={X_proc_b.iloc[0, j]:.6f} "
                      f"inf={X_proc_i.iloc[0, j]:.6f} diff={d:.6f}")


def _section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
