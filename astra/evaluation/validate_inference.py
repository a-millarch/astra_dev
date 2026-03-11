"""
Validate that training dataloader and inference pipeline produce matching predictions.

Spot-checks a specific holdout patient at multiple timeframes to verify that
normalization, padding, and model forward pass are consistent between both paths.

Usage::

    from astra.data.caching import prepare_data_and_dls_cached
    from astra.inference.pipeline import InferenceSession
    from astra.evaluation.validate_inference import validate_pipeline_consistency

    data = prepare_data_and_dls_cached(cfg)
    session = InferenceSession.load("my_model")

    results = validate_pipeline_consistency(
        data, session, pid='ABC123', timeframes=['1h', '1D', '3D'],
    )
"""

import logging
import re
from typing import Any, List, Optional

import numpy as np
import torch

from astra.data.dataloader import get_trajectory_lengths
from astra.evaluation.utils import time_to_step, step_to_time, time_to_hours
from astra.inference.pipeline import InferenceSession, extract_patient_from_data

logger = logging.getLogger(__name__)

DEFAULT_TIMEFRAMES = ["1h", "6h", "1D", "3D", "7D"]


def _parse_timeframe(tf: str):
    """Parse '1h', '6H', '3D', '30min' → (numeric_value, unit)."""
    tf = tf.strip().lower()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(h|d|min)$", tf)
    if not m:
        raise ValueError(
            f"Cannot parse timeframe '{tf}'. Use format like '1h', '3D', '30min'."
        )
    value = float(m.group(1))
    unit = m.group(2).upper() if m.group(2) == "d" else m.group(2)
    return value, unit


def _censor_single_sample(x_ts, x_ts_cat, step):
    """Zero out timesteps after *step* for a single sample.

    Works on numpy ``[c, seq]`` and torch ``[c, seq]`` arrays.
    Returns new (cloned/copied) objects; inputs are unchanged.
    """
    if isinstance(x_ts, torch.Tensor):
        x_ts = x_ts.clone()
        x_ts_cat = x_ts_cat.clone()
    else:
        x_ts = x_ts.copy()
        x_ts_cat = x_ts_cat.copy()

    x_ts[:, step + 1:] = 0.0
    x_ts_cat[:, step + 1:] = 0
    return x_ts, x_ts_cat


def validate_pipeline_consistency(
    data: dict,
    session: InferenceSession,
    pid: Any,
    timeframes: Optional[List[str]] = None,
    atol: float = 0.01,
) -> dict:
    """
    Compare predictions between the training dataloader pipeline and the
    inference pipeline for a specific holdout patient at multiple timeframes.

    Args:
        data: Dict from ``prepare_data_and_dls()`` or cache.
        session: ``InferenceSession`` with loaded model + bundle.
        pid: Patient PID as used in ``data["holdout"].tab_df``.
        timeframes: List of time offsets from admission, e.g. ``['1h', '1D', '3D']``.
            Defaults to ``['1h', '6h', '1D', '3D', '7D']``.
        atol: Absolute tolerance for flagging prediction discrepancies.

    Returns:
        Dict with keys ``'normalization'``, ``'timeframes'``, ``'all_passed'``.
    """
    if timeframes is None:
        timeframes = DEFAULT_TIMEFRAMES

    device = session.device
    model = session.model
    data_config = session.bundle.get("data_config")

    # ================================================================
    # 1. Locate patient in holdout
    # ================================================================
    holdout_pids = data["holdout"].tab_df["PID"].tolist()
    if pid not in holdout_pids:
        raise ValueError(
            f"PID {pid} not found in holdout. First 10: {holdout_pids[:10]}"
        )

    sample_idx = holdout_pids.index(pid)
    y_true = data["ty"][sample_idx]
    # Check if patient is deceased (mask_mortality applies in batch only)
    holdout_base = data["holdout"].base
    is_deceased = False
    if 'DOD' in holdout_base.columns:
        dod = holdout_base.loc[holdout_base.PID == pid, 'DOD']
        is_deceased = bool(dod.notna().any())

    logger.info(
        "Validating PID=%s (holdout idx=%d, y=%s, deceased=%s)",
        pid, sample_idx, y_true, is_deceased,
    )

    # ================================================================
    # 2. Training pipeline: pre-processed tensors from holdout dataset
    # ================================================================
    holdout_ds = data["holdout_mixed_dls"]._train_ds
    (
        x_ts_train,
        (x_cat_train, x_cont_train),
        x_ts_cat_train,
        traj_len_train,
    ), y_train = holdout_ds[sample_idx]

    traj_len_train_int = int(traj_len_train)

    # ================================================================
    # 3. Inference pipeline: raw tensors → _prepare_tensors
    # ================================================================
    patient_raw = extract_patient_from_data(data, pid)
    x_ts_raw = patient_raw["x_ts"]  # [c_in, seq_len]
    x_ts_cat_raw = patient_raw["x_ts_cat"]  # [n_cat_dims, seq_len]
    tab_df = patient_raw["tab_df"]

    x_ts_inf, x_cat_inf, x_cont_inf, x_ts_cat_inf, traj_len_inf = (
        session._prepare_tensors(
            x_ts_raw.copy(), x_ts_cat_raw.copy(), tab_df.copy()
        )
    )

    # ================================================================
    # 4. Compare normalization
    # ================================================================
    x_ts_train_np = x_ts_train.numpy()  # [c_in, seq_len]
    x_ts_inf_np = x_ts_inf.cpu().numpy()[0]  # [c_in, seq_len]

    traj = min(traj_len_train_int, traj_len_inf)
    norm_diff = np.abs(x_ts_train_np - x_ts_inf_np)

    channel_names = session.bundle.get("ts_channel_names", [])
    channel_diffs = {}
    for ch_idx, ch_name in enumerate(channel_names):
        if ch_idx >= norm_diff.shape[0]:
            break
        ch_diff = norm_diff[ch_idx, :traj]
        channel_diffs[ch_name] = {
            "max_diff": float(ch_diff.max()) if len(ch_diff) > 0 else 0.0,
            "mean_diff": float(ch_diff.mean()) if len(ch_diff) > 0 else 0.0,
        }

    # Tabular comparison
    x_cat_match = torch.equal(x_cat_train, x_cat_inf.cpu().squeeze(0))
    cont_diff = float(
        (x_cont_train - x_cont_inf.cpu().squeeze(0)).abs().max()
    ) if x_cont_train.numel() > 0 else 0.0

    # Trajectory length mismatch is expected for deceased patients:
    # mask_mortality shortens batch trajectories but is intentionally
    # NOT applied in inference (real-world: DOD unknown at inference time).
    traj_len_diff = traj_len_inf - traj_len_train_int
    traj_len_diff_expected = is_deceased and traj_len_diff > 0

    if traj_len_diff_expected:
        logger.info(
            "Trajectory length diff=%d (batch=%d, inf=%d) — expected for "
            "deceased patient (mask_mortality in batch only)",
            traj_len_diff, traj_len_train_int, traj_len_inf,
        )

    norm_result = {
        "traj_len_training": traj_len_train_int,
        "traj_len_inference": traj_len_inf,
        "traj_len_match": traj_len_train_int == traj_len_inf,
        "is_deceased": is_deceased,
        "traj_len_diff_expected": traj_len_diff_expected,
        "ts_max_diff": float(norm_diff[:, :traj].max()) if traj > 0 else 0.0,
        "ts_mean_diff": float(norm_diff[:, :traj].mean()) if traj > 0 else 0.0,
        "tab_cat_match": x_cat_match,
        "tab_cont_max_diff": cont_diff,
        "channel_diffs": channel_diffs,
    }

    # ================================================================
    # 5. Compare predictions at each timeframe
    # ================================================================
    tf_results = {}
    all_pass = True

    # --- Training pipeline: full-trajectory forward pass ---
    with torch.no_grad():
        x_ts_b = x_ts_train.unsqueeze(0).float().to(device)
        x_cat_b = x_cat_train.unsqueeze(0).to(device)
        x_cont_b = x_cont_train.unsqueeze(0).float().to(device)
        x_ts_cat_b = x_ts_cat_train.unsqueeze(0).float().to(device)
        traj_b = traj_len_train.unsqueeze(0).to(device)

        logits_train = model(
            (x_ts_b, (x_cat_b, x_cont_b), x_ts_cat_b, traj_b)
        )

    if session.is_temporal:
        probs_all_train = torch.sigmoid(logits_train).cpu().numpy()[0]
        prob_train_full = float(probs_all_train[traj_len_train_int - 1])
    else:
        probs_std = torch.softmax(logits_train, dim=1).cpu().numpy()[0]
        prob_train_full = float(probs_std[1])

    # --- Full-trajectory baseline (always included) ---
    inf_result_full = session.predict(
        x_ts_raw.copy(), x_ts_cat_raw.copy(), tab_df.copy(), pid=pid,
    )
    abs_diff = abs(prob_train_full - inf_result_full.probability)
    passed = abs_diff <= atol
    if not passed:
        all_pass = False
    tf_results["full_trajectory"] = {
        "step": traj_len_train_int - 1,
        "time_label": "full",
        "training_prob": prob_train_full,
        "inference_prob": inf_result_full.probability,
        "abs_diff": abs_diff,
        "passed": passed,
    }

    # --- Per-timeframe comparison ---
    for tf_str in timeframes:
        value, unit = _parse_timeframe(tf_str)
        step = time_to_step(value, unit, data_config=data_config)

        if step is None:
            tf_results[tf_str] = {
                "step": None,
                "skipped": True,
                "reason": "beyond bin grid",
            }
            continue

        if step >= traj_len_train_int:
            tf_results[tf_str] = {
                "step": step,
                "skipped": True,
                "reason": f"step {step} >= traj_len {traj_len_train_int}",
            }
            continue

        # Training path
        if session.is_temporal:
            prob_train = float(probs_all_train[step])
        else:
            # Censor normalized tensors and re-run forward pass
            x_ts_cens, x_ts_cat_cens = _censor_single_sample(
                x_ts_train, x_ts_cat_train, step
            )
            with torch.no_grad():
                traj_c = torch.tensor(
                    [min(traj_len_train_int, step + 1)],
                    dtype=torch.long, device=device,
                )
                logits_cens = model((
                    x_ts_cens.unsqueeze(0).float().to(device),
                    (x_cat_b, x_cont_b),
                    x_ts_cat_cens.unsqueeze(0).float().to(device),
                    traj_c,
                ))
            prob_train = float(
                torch.softmax(logits_cens, dim=1).cpu().numpy()[0, 1]
            )

        # Inference path
        if session.is_temporal:
            inf_result = session.predict(
                x_ts_raw.copy(), x_ts_cat_raw.copy(), tab_df.copy(),
                censor_step=step, pid=pid,
            )
        else:
            # Censor raw data; _prepare_tensors auto-detects shorter trajectory
            x_ts_raw_cens, x_ts_cat_raw_cens = _censor_single_sample(
                x_ts_raw, x_ts_cat_raw, step
            )
            inf_result = session.predict(
                x_ts_raw_cens, x_ts_cat_raw_cens, tab_df.copy(), pid=pid,
            )

        prob_inf = inf_result.probability
        abs_diff = abs(prob_train - prob_inf)
        passed = abs_diff <= atol
        if not passed:
            all_pass = False

        time_min = step_to_time(step, data_config=data_config)
        tf_results[tf_str] = {
            "step": step,
            "time_label": time_to_hours(time_min) if time_min else tf_str,
            "training_prob": prob_train,
            "inference_prob": prob_inf,
            "abs_diff": abs_diff,
            "passed": passed,
        }

    # ================================================================
    # 6. Summary
    # ================================================================
    norm_ok = (
        norm_result["traj_len_match"]
        and norm_result["ts_max_diff"] < atol
        and norm_result["tab_cat_match"]
        and norm_result["tab_cont_max_diff"] < atol
    )

    result = {
        "pid": pid,
        "y_true": int(y_true),
        "normalization": norm_result,
        "timeframes": tf_results,
        "all_passed": all_pass and norm_ok,
    }

    _print_report(result, atol)
    return result


def validate_multiple(
    data: dict,
    session: InferenceSession,
    n_patients: int = 5,
    timeframes: Optional[List[str]] = None,
    atol: float = 0.01,
) -> List[dict]:
    """Run validation on ``n_patients`` randomly sampled holdout patients.

    Returns:
        List of per-patient result dicts.
    """
    holdout_pids = data["holdout"].tab_df["PID"].tolist()
    rng = np.random.RandomState(42)
    sample_pids = rng.choice(
        holdout_pids, size=min(n_patients, len(holdout_pids)), replace=False,
    )

    results = []
    for pid in sample_pids:
        res = validate_pipeline_consistency(
            data, session, pid, timeframes=timeframes, atol=atol,
        )
        results.append(res)

    n_pass = sum(r["all_passed"] for r in results)
    print(f"\nOverall: {n_pass}/{len(results)} patients passed (atol={atol})")
    return results


# ====================================================================
# Pretty-print
# ====================================================================

def _print_report(result: dict, atol: float):
    pid = result["pid"]
    norm = result["normalization"]

    print(f"\n{'=' * 65}")
    print(f" Pipeline Consistency: PID={pid}  y={result['y_true']}")
    print(f"{'=' * 65}")

    tl_tag = "OK" if norm["traj_len_match"] else "MISMATCH"
    cat_tag = "OK" if norm["tab_cat_match"] else "MISMATCH"
    print(f"\n  Normalization")
    print(
        f"    Trajectory length : training={norm['traj_len_training']}, "
        f"inference={norm['traj_len_inference']}  [{tl_tag}]"
    )
    print(
        f"    TS tensor diff    : max={norm['ts_max_diff']:.6f}, "
        f"mean={norm['ts_mean_diff']:.6f}"
    )
    print(f"    Tabular cat codes : [{cat_tag}]")
    print(f"    Tabular cont diff : max={norm['tab_cont_max_diff']:.6f}")

    # Top differing channels
    sorted_ch = sorted(
        norm["channel_diffs"].items(),
        key=lambda x: x[1]["max_diff"],
        reverse=True,
    )
    top = [(n, d) for n, d in sorted_ch if d["max_diff"] > 1e-6]
    if top:
        print(f"    Top differing channels:")
        for ch_name, d in top[:5]:
            print(
                f"      {ch_name:<30s}  max={d['max_diff']:.6f}  "
                f"mean={d['mean_diff']:.6f}"
            )

    # Predictions
    print(f"\n  Predictions (atol={atol})")
    print(
        f"    {'Timeframe':<12} {'Step':>5}  "
        f"{'Training':>10}  {'Inference':>10}  {'Diff':>10}  {'Status':>6}"
    )
    print(f"    {'-' * 59}")

    for tf_str, tf in result["timeframes"].items():
        if tf.get("skipped"):
            reason = tf.get("reason", "")
            print(
                f"    {tf_str:<12} {str(tf.get('step', '?')):>5}  "
                f"{'--':>10}  {'--':>10}  {'--':>10}  {'SKIP':>6}  {reason}"
            )
            continue
        tag = "OK" if tf["passed"] else "FAIL"
        print(
            f"    {tf_str:<12} {tf['step']:>5}  "
            f"{tf['training_prob']:>10.6f}  {tf['inference_prob']:>10.6f}  "
            f"{tf['abs_diff']:>10.6f}  {tag:>6}"
        )

    verdict = "PASS" if result["all_passed"] else "FAIL"
    print(f"\n  Verdict: {verdict}")
    print(f"{'=' * 65}")
