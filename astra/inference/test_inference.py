"""
End-to-end test for the single-patient inference pipeline.

Run on Azure ML where data and models are available:
    python -m astra.inference.test_inference [--model-name MODEL] [--pid PID]

Tests:
1. Load deployment bundle and model
2. Extract a holdout patient
3. Run predict() and compare against batch evaluation
4. Run explain() and verify SHAP output structure
"""

import argparse
import logging
import sys

import numpy as np
import torch

from astra.utils import cfg

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Test single-patient inference pipeline')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Model name (default: cfg["model_name"])')
    parser.add_argument('--pid', type=int, default=None,
                        help='Patient ID to test (default: first holdout patient)')
    parser.add_argument('--skip-shap', action='store_true',
                        help='Skip SHAP computation (faster)')
    parser.add_argument('--save-bundle', action='store_true',
                        help='Save deployment bundle from cached data (run before first test)')
    parser.add_argument('--device', type=str, default=None,
                        help='Force device (default: auto-detect)')
    return parser.parse_args()


def save_bundle_from_cache(model_name):
    """Save deployment bundle using cached training data."""
    from astra.data.caching import prepare_data_and_dls_cached
    from astra.data.dataloader import save_deployment_bundle

    logger.info("Loading cached data...")
    data = prepare_data_and_dls_cached(cfg)
    logger.info("Saving deployment bundle...")
    path = save_deployment_bundle(data, cfg, model_name)
    logger.info(f"Bundle saved: {path}")
    return data


def test_prediction(session, patient, data):
    """Test predict() and compare against batch forward pass."""
    from astra.inference.pipeline import extract_patient_from_data

    logger.info("--- Testing prediction ---")
    result = session.predict(
        patient['x_ts'], patient['x_ts_cat'], patient['tab_df'],
        pid=patient['pid']
    )

    logger.info(f"  PID: {result.pid}")
    logger.info(f"  Probability: {result.probability:.4f}")
    logger.info(f"  Trajectory length: {result.trajectory_length}")
    logger.info(f"  True label: {patient['y']}")

    if session.is_temporal:
        logger.info(f"  Censor step: {result.censor_step}")
        if result.predictions_over_time is not None:
            valid = result.predictions_over_time[:result.trajectory_length]
            logger.info(f"  Prediction range: [{valid.min():.4f}, {valid.max():.4f}]")

    # Sanity check: probability should be in [0, 1]
    assert 0.0 <= result.probability <= 1.0, f"Probability out of range: {result.probability}"
    logger.info("  [PASS] Probability in valid range")

    # Cross-check: run model directly on the same tensors
    x_ts_t, x_cat_t, x_cont_t, x_ts_cat_t, traj_len = session._prepare_tensors(
        patient['x_ts'], patient['x_ts_cat'], patient['tab_df']
    )
    with torch.no_grad():
        logits_direct = session.model((x_ts_t, (x_cat_t, x_cont_t), x_ts_cat_t))

    if session.is_temporal:
        step = result.censor_step
        prob_direct = float(torch.sigmoid(logits_direct[0, step]).cpu())
    else:
        prob_direct = float(torch.softmax(logits_direct, dim=1)[0, 1].cpu())

    diff = abs(result.probability - prob_direct)
    logger.info(f"  Direct forward prob: {prob_direct:.6f} (diff: {diff:.2e})")
    assert diff < 1e-5, f"Prediction mismatch: predict()={result.probability}, direct={prob_direct}"
    logger.info("  [PASS] predict() matches direct forward pass")

    return result


def test_shap(session, patient):
    """Test explain() and verify output structure."""
    logger.info("--- Testing SHAP explanation ---")

    shap_result = session.explain(
        patient['x_ts'], patient['x_ts_cat'], patient['tab_df'],
        pid=patient['pid']
    )

    # Verify structure
    assert isinstance(shap_result.ts_shap, dict), "ts_shap should be a dict"
    n_channels = len(shap_result.ts_shap)
    logger.info(f"  TS SHAP channels: {n_channels}")

    first_key = list(shap_result.ts_shap.keys())[0]
    first_shape = shap_result.ts_shap[first_key].shape
    logger.info(f"  First channel shape: {first_shape}")

    if shap_result.cat_ts_shap:
        logger.info(f"  Categorical TS categories: {len(shap_result.cat_ts_shap)}")
    if shap_result.static_cat_shap:
        logger.info(f"  Static categorical features: {shap_result.static_cat_shap}")
    if shap_result.static_cont_shap:
        logger.info(f"  Static continuous features: {shap_result.static_cont_shap}")

    logger.info(f"  Top 10 features by importance:")
    for name, imp in shap_result.top_features[:10]:
        logger.info(f"    {name}: {imp:.6f}")

    # Verify no NaN in SHAP values
    for name, arr in shap_result.ts_shap.items():
        assert not np.isnan(arr).any(), f"NaN in ts_shap[{name}]"
    logger.info("  [PASS] No NaN in SHAP values")

    return shap_result


def main():
    args = parse_args()
    model_name = args.model_name or cfg["model_name"]

    if args.save_bundle:
        data = save_bundle_from_cache(model_name)
    else:
        data = None

    # Load data for patient extraction (if not already loaded)
    if data is None:
        from astra.data.caching import prepare_data_and_dls_cached
        logger.info("Loading cached data for patient extraction...")
        data = prepare_data_and_dls_cached(cfg)

    # Load inference session
    from astra.inference.pipeline import InferenceSession, extract_patient_from_data

    logger.info(f"Loading inference session for model '{model_name}'...")
    session = InferenceSession.load(model_name, device=args.device)
    logger.info(f"  Temporal head: {session.is_temporal}")
    logger.info(f"  Channels: {len(session.bundle['ts_channel_names'])}")
    logger.info(f"  SHAP background: {session._bg['ts'].shape[0]} samples")

    # Extract test patient
    holdout_pids = data["holdout"].tab_df['PID'].tolist()
    pid = args.pid or holdout_pids[0]
    logger.info(f"Extracting patient PID={pid}...")
    patient = extract_patient_from_data(data, pid)
    logger.info(f"  x_ts shape: {patient['x_ts'].shape}")
    logger.info(f"  x_ts_cat shape: {patient['x_ts_cat'].shape}")
    logger.info(f"  Target: {patient['y']}")

    # Test prediction
    pred_result = test_prediction(session, patient, data)

    # Test SHAP
    if not args.skip_shap:
        shap_result = test_shap(session, patient)
    else:
        logger.info("--- Skipping SHAP (--skip-shap) ---")

    logger.info("=== All tests passed ===")


if __name__ == "__main__":
    main()
