"""Azure-side validation driver for the handoff-clean inference API.

Run from the repo root in the secure environment (real data + model
artifacts present). Complements the synthetic checks that already run
anywhere (`pytest tests/`, `export_artifacts validate --self-test`).

    python scripts/azure_handoff_check.py --config configs/defaults.yaml \
        [--model-name <M>]                      # default: model_name from --config
        [--n-patients 3] [--hours 24] [--skip-shap] [--skip-selftest] \
        [--skip-cli] [--export-dir handoff_check] \
        [--cpr-hash HASH --service-date DATE]   # explicit patient instead of auto-pick

Config-first: the chosen config supplies the model name, the base_df path for
patient auto-picking, and the data-prep settings used when building patient
contexts; the same config file is shipped in the exported bundle.

Checks (each reported PASS/FAIL):
  1. self-test        — synthetic export/validate round trip (tiny model)
  2. real export      — export the real model bundle + validate hashes/load/forward
  3. parity (xN)      — AstraPredictor.predict/.explain vs the established
                        SimulationRunner + InferenceSession path on holdout
                        patients: probabilities/curves must match to 1e-6;
                        SHAP arrays exact or correlation >= 0.95
                        (GradientExplainer sampling can add small noise)
  4. CLI smoke        — python -m astra.inference.run_inference end-to-end JSON

Exit code 0 only if every executed check passes.
"""

import argparse
import json
import logging
import logging.handlers
import os
import subprocess
import sys
import tempfile
import traceback
import warnings

import numpy as np
import pandas as pd

# Under the 'astra' hierarchy so setup_logging()'s console + file handlers
# apply (a bare __main__ logger has no handler and its messages vanish).
logger = logging.getLogger('astra.scripts.handoff_check')

ATOL = 1e-6
SHAP_CORR_THRESHOLD = 0.95


def _pick_patients(n, hours, cfg):
    """Most recent patients (holdout-ish) with trajectories longer than *hours*."""
    from astra.utils import get_base_df

    base = get_base_df(cfg.get('base_df_path'))
    dur_h = (pd.to_datetime(base['end']) - pd.to_datetime(base['start'])
             ).dt.total_seconds() / 3600.0
    eligible = base[dur_h >= hours + 1.0].sort_values('start', ascending=False)
    if eligible.empty:
        raise RuntimeError(
            f'No patients with trajectories >= {hours + 1.0:.0f}h in base_df')
    picks = [(row['CPR_hash'], str(row['ServiceDate']))
             for _, row in eligible.head(n).iterrows()]
    # Full identifiers, so any patient can be re-targeted with
    # --cpr-hash/--service-date (summary rows only show the 8-char prefix).
    for cpr_hash, service_date in picks:
        logger.info('Picked patient: --cpr-hash %s --service-date "%s"',
                    cpr_hash, service_date)
    return picks


def _describe_exception(exc) -> str:
    """`TypeName(msg) at file:line (in func)` — enough to localize a crash
    from the summary table alone (full traceback goes to the log)."""
    tb = traceback.extract_tb(exc.__traceback__)
    where = ''
    if tb:
        frame = tb[-1]
        where = f' at {os.path.basename(frame.filename)}:{frame.lineno} (in {frame.name})'
    return f'{type(exc).__name__}({exc}){where}'[:160]


def _compare_curves(name, facade_probs, ref_probs, rows):
    """NaN-aware elementwise comparison at ATOL."""
    a = np.asarray([np.nan if p is None else p for p in facade_probs], dtype=float)
    b = np.asarray(ref_probs, dtype=float)[: len(a)]
    if len(a) != len(b):
        rows.append((name, 'FAIL', f'length {len(a)} vs {len(b)}'))
        return
    both = np.isfinite(a) & np.isfinite(b)
    nan_mismatch = int(np.sum(np.isfinite(a) != np.isfinite(b)))
    max_diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
    if nan_mismatch == 0 and max_diff <= ATOL:
        rows.append((name, 'PASS', f'max|diff|={max_diff:.2e} over {int(both.sum())} steps'))
    else:
        rows.append((name, 'FAIL',
                     f'max|diff|={max_diff:.2e}, NaN-pattern mismatches={nan_mismatch}'))


def _compare_shap(name, facade_arr, ref_arr, rows):
    """Exact match at ATOL, else Pearson correlation (explainer sampling noise)."""
    a = np.asarray(facade_arr, dtype=float).ravel()
    b = np.asarray(ref_arr, dtype=float).ravel()
    if a.shape != b.shape:
        rows.append((name, 'FAIL', f'shape {a.shape} vs {b.shape}'))
        return
    a = np.nan_to_num(a)
    b = np.nan_to_num(b)
    max_diff = float(np.max(np.abs(a - b))) if len(a) else 0.0
    if max_diff <= ATOL:
        rows.append((name, 'PASS', f'exact (max|diff|={max_diff:.2e})'))
        return
    if np.std(a) == 0 or np.std(b) == 0:
        rows.append((name, 'FAIL', f'constant array, max|diff|={max_diff:.2e}'))
        return
    corr = float(np.corrcoef(a, b)[0, 1])
    status = 'PASS' if corr >= SHAP_CORR_THRESHOLD else 'FAIL'
    rows.append((name, status, f'corr={corr:.4f}, max|diff|={max_diff:.2e} '
                               f'(explainer sampling tolerance)'))


def check_parity(model_name, config_path, cfg, cpr_hash, service_date,
                 hours, skip_shap, rows):
    """Facade vs direct InferenceSession/SimulationRunner on one patient."""
    from astra.inference import AstraPredictor, InferenceSession, SimulationRunner

    tag = f'parity {str(cpr_hash)[:8]}@{hours:g}h'
    logger.info('=== %s (service_date=%s) ===', tag, service_date)

    # --- Reference: the established dashboard path ---
    session = InferenceSession.load(model_name, device='cpu')
    runner = SimulationRunner(session)
    runner.setup(cpr_hash=cpr_hash, service_date=service_date, cfg=cfg)
    runner.advance_to(hours=hours)
    ctx_ref = runner.context
    ref = session.predict_from_context(ctx_ref)

    # --- Facade ---
    predictor = AstraPredictor.load(model_name, config_path=config_path,
                                    device='cpu')
    ts = ctx_ref.admission_time + pd.Timedelta(hours=hours)
    resp = predictor.predict(cpr_hash, ts, service_date)

    p_diff = abs(resp.probability - float(ref.probability))
    rows.append((f'{tag} probability',
                 'PASS' if p_diff <= ATOL else 'FAIL',
                 f'facade={resp.probability:.6f} ref={ref.probability:.6f} '
                 f'diff={p_diff:.2e}'))

    if resp.curve is not None:
        if session.is_temporal and ref.predictions_over_time is not None:
            ref_curve = np.asarray(ref.predictions_over_time)[:resp.trajectory_length]
        else:
            ref_curve = np.asarray(runner._prediction_curve)[:resp.trajectory_length]
        _compare_curves(f'{tag} curve', resp.curve.probabilities, ref_curve, rows)

    if not skip_shap:
        ref_shap = session.explain_from_context(ctx_ref)
        ref_dict, _, _, _ = session.shap_to_viz_dict(
            ref_shap, x_ts=ctx_ref.x_ts, x_ts_cat=ctx_ref.x_ts_cat,
            tab_df=ctx_ref.tab_df)
        expl = predictor.explain(cpr_hash, ts, service_date)

        if list(expl.channels) != list(session.bundle['ts_channel_names']):
            rows.append((f'{tag} shap channels', 'FAIL', 'channel order mismatch'))
        else:
            rows.append((f'{tag} shap channels', 'PASS',
                         f'{len(expl.channels)} channels, order matches bundle'))
        _compare_shap(f'{tag} ts_shap', expl.ts_shap, ref_dict['ts_shap'][0], rows)

        ref_steps = int(ref_dict['trajectory_length'])
        got_steps = int(expl.trajectory_length)
        rows.append((f'{tag} shap traj_len',
                     'PASS' if got_steps == ref_steps else 'FAIL',
                     f'facade={got_steps} ref={ref_steps}'))


def check_cli(config_path, cpr_hash, service_date, hours, rows):
    with tempfile.TemporaryDirectory() as tmp:
        out_json = os.path.join(tmp, 'result.json')
        # Absolute timestamp not needed: service_date + hours margin is enough
        # for a smoke test; use service_date + hours as the eval time.
        ts = (pd.Timestamp(service_date) + pd.Timedelta(hours=hours)).isoformat()
        cmd = [sys.executable, '-m', 'astra.inference.run_inference',
               '--config', config_path,
               '--patient-id', str(cpr_hash),
               '--service-date', str(service_date),
               '--timestamp', ts,
               '--device', 'cpu',   # keep the whole check GPU-independent
               '--out', out_json]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or '')[-400:]
            rows.append(('CLI smoke', 'FAIL', f'exit {proc.returncode}: ...{tail}'))
            return
        with open(out_json, encoding='utf-8') as fh:
            payload = json.load(fh)
        prob = payload['prediction']['probability']
        ok = isinstance(prob, (int, float)) and 0.0 <= prob <= 1.0
        rows.append(('CLI smoke', 'PASS' if ok else 'FAIL',
                     f'P={prob} written to JSON'))


def print_summary(rows):
    width = min(max((len(n) for n, _, _ in rows), default=24), 60)
    line = '=' * (width + 50)
    print()
    print(line)
    print('AZURE HANDOFF CHECK SUMMARY')
    print('-' * (width + 50))
    for name, status, detail in rows:
        print(f'{status:<5} {name:<{width}}  {detail}')
    n_fail = sum(1 for _, s, _ in rows if s == 'FAIL')
    print('-' * (width + 50))
    print(f"RESULT: {'FAIL' if n_fail else 'PASS'} "
          f'({sum(1 for _, s, _ in rows if s == "PASS")} passed, {n_fail} failed)')
    print(line)
    return 1 if n_fail else 0


def main():
    parser = argparse.ArgumentParser(
        description='Azure-side validation of the handoff inference API.')
    parser.add_argument('--config', default='configs/defaults.yaml',
                        help='Config YAML — supplies model_name, base_df path '
                             'and data-prep settings; shipped in the export '
                             '(default: configs/defaults.yaml)')
    parser.add_argument('--model-name', default=None,
                        help='Override the model_name from --config')
    parser.add_argument('--n-patients', type=int, default=3,
                        help='Holdout patients to auto-pick for parity (default 3)')
    parser.add_argument('--hours', type=float, default=24.0,
                        help='Evaluation time in hours after admission (default 24)')
    parser.add_argument('--cpr-hash', default=None,
                        help='Explicit patient (with --service-date) instead of auto-pick')
    parser.add_argument('--service-date', default=None)
    parser.add_argument('--export-dir', default='handoff_check',
                        help='Directory for the real-model export round trip')
    parser.add_argument('--sign-off', default=None,
                        help='Recorded in the exported manifest (see export_artifacts)')
    parser.add_argument('--skip-selftest', action='store_true')
    parser.add_argument('--skip-shap', action='store_true')
    parser.add_argument('--skip-cli', action='store_true')
    parser.add_argument('--verbose', action='store_true',
                        help='Keep full INFO console output during parity '
                             '(default: console quiets to WARNING for the '
                             'stepping loops; everything still goes to the '
                             'file log)')
    args = parser.parse_args()

    from astra.utils import setup_logging, get_cfg
    setup_logging(level=logging.INFO)

    cfg = get_cfg(args.config)
    model_name = args.model_name or cfg.get('model_name')
    if not model_name:
        parser.error(f"{args.config} has no 'model_name' key — pass --model-name")
    logger.info('Config: %s -> model_name=%r', args.config, model_name)

    rows = []

    # 1. Synthetic self-test
    if not args.skip_selftest:
        from astra.inference.export_artifacts import run_self_test
        rc = run_self_test(explain_smoke=not args.skip_shap)
        rows.append(('export self-test', 'PASS' if rc == 0 else 'FAIL',
                     'synthetic round trip'))

    # 2. Real-model export + validate round trip (ships the chosen config)
    try:
        from astra.inference.export_artifacts import run_export, run_validate
        run_export(model_name, out_dir=args.export_dir,
                   config_path=args.config, sign_off=args.sign_off)
        rc = run_validate(args.export_dir, model_name=model_name,
                          explain_smoke=not args.skip_shap)
        rows.append(('real-model export+validate', 'PASS' if rc == 0 else 'FAIL',
                     args.export_dir))
    except Exception as exc:
        logger.exception('Real-model export/validate crashed')
        rows.append(('real-model export+validate', 'FAIL', _describe_exception(exc)))

    # 3. Golden-patient parity
    if args.cpr_hash:
        if not args.service_date:
            parser.error('--cpr-hash requires --service-date')
        patients = [(args.cpr_hash, args.service_date)]
    else:
        try:
            patients = _pick_patients(args.n_patients, args.hours, cfg)
        except Exception as exc:
            logger.exception('Patient auto-pick failed')
            rows.append(('patient auto-pick', 'FAIL', _describe_exception(exc)))
            patients = []

    # The simulation stepping logs one INFO line per bin (x patients) and
    # pandas emits repeated FutureWarnings — enough to overflow terminals.
    # Quiet the console; the file log (logging/astra.log) keeps full DEBUG.
    if not args.verbose:
        warnings.filterwarnings('ignore', category=FutureWarning)
        for h in logging.getLogger('astra').handlers:
            if not isinstance(h, logging.handlers.TimedRotatingFileHandler):
                h.setLevel(logging.WARNING)
        print('(console quieted for parity stepping — full detail in the '
              'file log; use --verbose to keep it)')

    for cpr_hash, service_date in patients:
        print(f'Parity: {str(cpr_hash)[:8]}... @ {args.hours:g}h '
              f'(service_date={service_date}) ...')
        try:
            check_parity(model_name, args.config, cfg, cpr_hash, service_date,
                         args.hours, args.skip_shap, rows)
        except Exception as exc:
            logger.exception('Parity check crashed for %s', str(cpr_hash)[:8])
            rows.append((f'parity {str(cpr_hash)[:8]}', 'FAIL',
                         _describe_exception(exc)))

    # 4. CLI smoke (first patient)
    if not args.skip_cli and patients:
        try:
            check_cli(args.config, patients[0][0], patients[0][1],
                      args.hours, rows)
        except Exception as exc:
            logger.exception('CLI smoke crashed')
            rows.append(('CLI smoke', 'FAIL', _describe_exception(exc)))

    return print_summary(rows)


if __name__ == '__main__':
    sys.exit(main())
