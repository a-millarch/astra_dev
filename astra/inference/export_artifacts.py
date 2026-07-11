"""
Export and validate ASTRA deployment artifact bundles for external handoff.

``export`` collects everything an external team needs to run inference in
their own environment (no access to our data), fingerprints every file with
SHA-256 and writes a ``manifest.json``:

    handoff/
      manifest.json
      models/deployment/deployment_{name}.pkl
      models/{name}.pth
      models/calibrators/{name}/**          (if present)
      models/ebm/*.pkl                      (if the model uses '_ebm_pred')
      configs/defaults.yaml
      data/external/metadata.csv            (required — per-concept datetime columns)
      docs/HANDOFF.md                       (if present)
      examples/synthetic_patient.json       (generated, fully synthetic)

``validate`` is run BY THE RECEIVING TEAM in their environment as an
acceptance test: hash check → load model → synthetic forward pass
(→ optional SHAP smoke test). Exit code 0 = PASS, 1 = FAIL.

Usage:
    python -m astra.inference.export_artifacts export --model-name X [--artifacts-dir models]
        [--out handoff] [--config configs/defaults.yaml] [--no-ebm] [--dry-run]
    python -m astra.inference.export_artifacts validate --dir handoff
        [--model-name X] [--explain-smoke]
    python -m astra.inference.export_artifacts validate --self-test
        # end-to-end smoke test on generated tiny artifacts — needs no real files

IMPORTANT: the deployment bundle's ``shap_background`` contains
patient-derived tensors. The ``shap_background.sign_off`` field in
manifest.json MUST be completed by data protection before the bundle
leaves the secure environment.
"""

import argparse
import hashlib
import json
import logging
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SHAP_BG_NOTE = ('patient-derived background tensors — requires data-protection '
                'sign-off before external transfer')


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _sha256(path):
    """SHA-256 hex digest of a file (chunked read)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _to_posix(rel_path):
    """Normalize a relative path to forward slashes for the manifest."""
    return str(rel_path).replace('\\', '/')


def _git_commit():
    """Current git commit hash, or None if unavailable."""
    try:
        out = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _collect_versions():
    """Runtime environment versions for the manifest."""
    import platform
    from importlib import metadata as importlib_metadata

    versions = {'python': platform.python_version()}
    for pkg in ('torch', 'numpy', 'pandas', 'shap'):
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except Exception:
            versions[pkg] = None
    return versions


def _jsonable(obj):
    """Recursively convert to JSON-serializable types (Timestamps → ISO str)."""
    import numpy as np
    import pandas as pd

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _print_shap_warning(sign_off=None):
    if sign_off:
        logger.info('shap_background sign-off recorded in manifest: %s', sign_off)
        print(f'shap_background sign-off recorded: {sign_off}')
        return
    lines = [
        '!' * 78,
        '!!  WARNING: PATIENT-DERIVED DATA IN BUNDLE',
        "!!  The deployment bundle's shap_background contains background tensors",
        '!!  derived from real patients. Data-protection sign-off is REQUIRED',
        '!!  before this bundle leaves the secure environment.',
        "!!  Complete manifest.json -> shap_background.sign_off (currently null),",
        "!!  or re-run export with --sign-off '<who approved, when, basis>'.",
        '!' * 78,
    ]
    banner = '\n'.join(lines)
    logger.warning('shap_background contains patient-derived data; '
                   'sign_off in manifest.json must be completed before sharing.')
    print(banner)


def write_minimal_support_files(dest_dir):
    """Write a minimal config + metadata.csv for self-test / hermetic tests.

    Real exports ship ``configs/defaults.yaml`` and
    ``data/external/metadata.csv`` from the repo; these placeholders let the
    tool run end-to-end where neither exists.

    Returns:
        Dict with keys 'config' and 'metadata_csv' (absolute paths).
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    config = dest / 'defaults.yaml'
    config.write_text(
        '# Minimal placeholder config generated for export self-test.\n'
        '# Real exports ship the project configs/defaults.yaml instead.\n'
        'model:\n'
        '  d_model: 16\n',
        encoding='utf-8',
    )

    metadata = dest / 'metadata.csv'
    metadata.write_text(
        'filename,dt_colname,value_colname,type_colname,ts_offset\n'
        'VitaleVaerdier,RECORDED_TIME,MEASUREMENT_VALUE,,0\n'
        'Medicin,TAKEN_TIME,ATC,,0\n',
        encoding='utf-8',
    )
    return {'config': str(config), 'metadata_csv': str(metadata)}


# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------

def run_export(model_name, artifacts_dir='models', out_dir='handoff',
               config_path='configs/defaults.yaml',
               metadata_csv='data/external/metadata.csv',
               handoff_doc='docs/HANDOFF.md',
               include_ebm=True, dry_run=False, sign_off=None):
    """Collect + fingerprint deployment artifacts into *out_dir*.

    Args:
        model_name: Model name (matches training save name).
        artifacts_dir: Source directory holding {name}.pth, deployment/,
            calibrators/, ebm/.
        out_dir: Destination handoff directory.
        config_path: Project config to ship (skipped with a warning if absent).
        metadata_csv: REQUIRED per-concept datetime-column definitions
            (error if missing).
        handoff_doc: Shipped if present.
        include_ebm: Copy models/ebm/*.pkl when the model uses '_ebm_pred'.
        dry_run: Report the copy plan; write nothing.
        sign_off: Data-protection approval to record in the manifest for the
            patient-derived shap_background (who approved, when, on what
            basis). When omitted the field is null and a warning is printed.

    Returns:
        The manifest dict (or a plan dict when ``dry_run=True``).
    """
    from astra.data.dataloader import load_deployment_bundle
    from astra.inference.synthetic import make_synthetic_raw_data

    artifacts_dir = Path(artifacts_dir)
    out_dir = Path(out_dir)
    bundle_dir = artifacts_dir / 'deployment'
    bundle_src = bundle_dir / f'deployment_{model_name}.pkl'
    weights_src = artifacts_dir / f'{model_name}.pth'

    if not bundle_src.is_file():
        raise FileNotFoundError(f'Deployment bundle not found: {bundle_src}')
    if not weights_src.is_file():
        raise FileNotFoundError(f'Model weights not found: {weights_src}')
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.is_file():
        raise FileNotFoundError(
            f'Required metadata file not found: {metadata_csv} — it defines '
            f'per-concept datetime columns and must ship with the bundle.'
        )

    bundle = load_deployment_bundle(model_name, str(bundle_dir))
    params = bundle['model_params']

    # ------------------------------------------------------------------
    # Build the copy plan: (absolute src, relative dest with forward slashes)
    # ------------------------------------------------------------------
    copies = [
        (bundle_src, f'models/deployment/deployment_{model_name}.pkl'),
        (weights_src, f'models/{model_name}.pth'),
    ]

    # Calibrators (optional)
    calibration_meta = None
    calib_dir = artifacts_dir / 'calibrators' / model_name
    if calib_dir.is_dir():
        for p in sorted(calib_dir.rglob('*')):
            if p.is_file():
                rel = Path('models/calibrators') / model_name / p.relative_to(calib_dir)
                copies.append((p, _to_posix(rel)))
        meta_path = calib_dir / 'metadata.json'
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                calibration_meta = {
                    'method': meta.get('best_method'),
                    'timepoints': meta.get('timepoints'),
                }
            except Exception as exc:
                logger.warning('Could not parse calibrator metadata %s: %s',
                               meta_path, exc)
    else:
        logger.info('No calibrators found at %s (optional).', calib_dir)

    # EBM models (only when the model consumes the '_ebm_pred' channel)
    ebm_files = []
    ebm_included = False
    has_ebm_channel = '_ebm_pred' in (bundle.get('ts_channel_names') or [])
    if has_ebm_channel and include_ebm:
        ebm_dir = artifacts_dir / 'ebm'
        pkls = sorted(ebm_dir.glob('*.pkl')) if ebm_dir.is_dir() else []
        if pkls:
            ebm_included = True
            for p in pkls:
                rel = f'models/ebm/{p.name}'
                copies.append((p, rel))
                ebm_files.append(rel)
        else:
            logger.warning("Model uses '_ebm_pred' but no EBM models found at %s "
                           '— inference will fail without them!', ebm_dir)
    elif has_ebm_channel and not include_ebm:
        logger.warning("--no-ebm: EBM models excluded although '_ebm_pred' is a "
                       'model input channel.')

    # Config (warn if missing), metadata (required, checked above), handoff doc
    config_path = Path(config_path)
    if config_path.is_file():
        copies.append((config_path, f'configs/{config_path.name}'))
    else:
        logger.warning('Config file not found: %s — not included.', config_path)

    copies.append((metadata_csv, f'data/external/{metadata_csv.name}'))

    handoff_doc = Path(handoff_doc)
    if handoff_doc.is_file():
        copies.append((handoff_doc, f'docs/{handoff_doc.name}'))
    else:
        logger.info('No handoff doc at %s (optional).', handoff_doc)

    example_rel = 'examples/synthetic_patient.json'

    # ------------------------------------------------------------------
    # Dry run: report only
    # ------------------------------------------------------------------
    if dry_run:
        logger.info('[dry-run] Export plan for model %r → %s', model_name, out_dir)
        for src, rel in copies:
            logger.info('[dry-run]   %s  ->  %s', src, rel)
        logger.info('[dry-run]   <generated synthetic patient>  ->  %s', example_rel)
        logger.info('[dry-run]   <generated manifest>  ->  manifest.json')
        print(f'[dry-run] Would export {len(copies) + 2} files to {out_dir} '
              f'(nothing written).')
        for src, rel in copies:
            print(f'[dry-run]   {rel}  <-  {src}')
        print(f'[dry-run]   {example_rel}  <-  <generated>')
        print(f'[dry-run]   manifest.json  <-  <generated>')
        _print_shap_warning()
        return {
            'dry_run': True,
            'model_name': model_name,
            'planned_files': [rel for _, rel in copies] + [example_rel, 'manifest.json'],
        }

    # ------------------------------------------------------------------
    # Copy files
    # ------------------------------------------------------------------
    for src, rel in copies:
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        logger.info('Copied %s -> %s', src, rel)

    # Synthetic example patient (generated from the bundle itself)
    raw_data = make_synthetic_raw_data(bundle)
    example_path = out_dir / example_rel
    example_path.parent.mkdir(parents=True, exist_ok=True)
    example_path.write_text(
        json.dumps(_jsonable(raw_data), indent=2), encoding='utf-8'
    )
    logger.info('Generated %s', example_rel)

    # ------------------------------------------------------------------
    # Fingerprint everything and write the manifest
    # ------------------------------------------------------------------
    file_entries = []
    for rel in [rel for _, rel in copies] + [example_rel]:
        p = out_dir / rel
        file_entries.append({
            'path': _to_posix(rel),
            'sha256': _sha256(p),
            'bytes': p.stat().st_size,
        })
    file_entries.sort(key=lambda e: e['path'])

    bg = bundle.get('shap_background')
    n_bg = int(bg['ts'].shape[0]) if isinstance(bg, dict) and 'ts' in bg else 0

    manifest = {
        'schema_version': SCHEMA_VERSION,
        'model_name': model_name,
        'created_at': datetime.now().isoformat(),
        'git_commit': _git_commit(),
        'versions': _collect_versions(),
        'model': {
            'temporal_head': bool(params.get('temporal_head', False)),
            'survival_mode': bool(params.get('survival_mode', False)),
            'seq_len': int(params['seq_len']),
            'c_in': int(params['c_in']),
            'd_model': int(params['d_model']),
            'n_layers': int(params['n_layers']),
            'head_pool': params.get('head_pool', 'flatten'),
        },
        'channels': list(bundle.get('ts_channel_names') or []),
        'static_features': {
            'continuous': list(bundle.get('tab_feature_names') or []),
            'categorical': list(bundle.get('cat_feature_names')
                                or params.get('classes', {}).keys()),
        },
        'bin_intervals': dict(bundle['data_config']['bin_intervals']),
        'bin_freq_include': list(bundle['data_config']['bin_freq_include']),
        'calibration': calibration_meta,
        'ebm': {'included': ebm_included, 'files': ebm_files},
        'shap_background': {
            'n_samples': n_bg,
            'note': SHAP_BG_NOTE,
            'sign_off': sign_off,
        },
        'files': file_entries,
    }

    manifest_path = out_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    logger.info('Wrote manifest with %d files: %s', len(file_entries), manifest_path)

    total_bytes = sum(e['bytes'] for e in file_entries)
    print(f'Exported {len(file_entries)} files ({total_bytes / 1e6:.2f} MB) '
          f'to {out_dir}')
    print(f'Manifest: {manifest_path}')
    _print_shap_warning(sign_off)
    return manifest


# ---------------------------------------------------------------------------
# VALIDATE
# ---------------------------------------------------------------------------

def _print_summary(rows):
    """Print the per-check PASS/FAIL summary table."""
    if not rows:
        return
    width = min(max(max(len(name) for name, _, _ in rows), 24), 70)
    line = '=' * (width + 40)
    print()
    print(line)
    print('VALIDATION SUMMARY')
    print('-' * (width + 40))
    for name, status, detail in rows:
        print(f'{status:<5} {name:<{width}}  {detail}')
    n_pass = sum(1 for _, s, _ in rows if s == 'PASS')
    n_fail = sum(1 for _, s, _ in rows if s == 'FAIL')
    n_skip = sum(1 for _, s, _ in rows if s == 'SKIP')
    print('-' * (width + 40))
    verdict = 'FAIL' if n_fail else 'PASS'
    print(f'RESULT: {verdict}  ({n_pass} passed, {n_fail} failed, {n_skip} skipped)')
    print(line)


def run_validate(handoff_dir, model_name=None, explain_smoke=False):
    """Acceptance test on an exported handoff directory.

    Checks, in order:
      1. manifest.json present; every listed file exists with matching
         size + SHA-256.
      2. ``InferenceSession.load`` succeeds on CPU.
      3. Bin-grid consistency: seq_len == get_total_steps(data_config).
      4. Synthetic patient forward pass: 0 <= probability <= 1; for temporal
         models, predictions_over_time has length seq_len.
      5. Optional (--explain-smoke): SHAP explanation returns non-empty
         ts_shap (SKIP if shap is not installed).

    Returns:
        0 on PASS, 1 on any FAIL.
    """
    handoff = Path(handoff_dir)
    rows = []

    def add(name, status, detail=''):
        rows.append((name, status, detail))
        log = logger.error if status == 'FAIL' else logger.info
        log('%s: %s %s', status, name, detail)

    # --- 1. Manifest + file integrity --------------------------------------
    manifest_path = handoff / 'manifest.json'
    if not manifest_path.is_file():
        add('manifest.json present', 'FAIL', f'not found at {manifest_path}')
        _print_summary(rows)
        return 1
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except Exception as exc:
        add('manifest.json present', 'FAIL', f'unreadable: {exc}')
        _print_summary(rows)
        return 1
    add('manifest.json present', 'PASS',
        f"schema_version={manifest.get('schema_version')}")

    model_name = model_name or manifest.get('model_name')

    integrity_ok = True
    for entry in manifest.get('files', []):
        rel = entry.get('path', '?')
        p = handoff / rel
        if not p.is_file():
            add(f'file {rel}', 'FAIL', 'missing')
            integrity_ok = False
            continue
        size = p.stat().st_size
        if size != entry.get('bytes'):
            add(f'file {rel}', 'FAIL',
                f"size mismatch: {size} != {entry.get('bytes')}")
            integrity_ok = False
            continue
        digest = _sha256(p)
        if digest != entry.get('sha256'):
            add(f'file {rel}', 'FAIL', 'sha256 mismatch')
            integrity_ok = False
            continue
        add(f'file {rel}', 'PASS', f'{size} bytes')

    if not integrity_ok:
        add('load model', 'SKIP', 'integrity check failed')
        add('synthetic forward pass', 'SKIP', 'integrity check failed')
        _print_summary(rows)
        return 1

    # --- 2. Load the model ---------------------------------------------------
    try:
        from astra.inference.pipeline import InferenceSession
        session = InferenceSession.load(
            model_name,
            device='cpu',
            bundle_dir=str(handoff / 'models' / 'deployment'),
            weights_dir=str(handoff / 'models'),
        )
        n_params = sum(p.numel() for p in session.model.parameters())
        add('load model', 'PASS',
            f'{n_params} parameters, temporal_head={session.is_temporal}')
    except Exception as exc:
        add('load model', 'FAIL', f'{type(exc).__name__}: {exc}')
        _print_summary(rows)
        return 1

    # --- 3. Bin-grid consistency ---------------------------------------------
    seq_len = session.bundle['model_params']['seq_len']
    try:
        from astra.evaluation.utils import get_total_steps
        data_config = session.bundle.get('data_config')
        if data_config is None:
            add('bin-grid consistency', 'FAIL', "bundle has no 'data_config'")
        else:
            expected = get_total_steps(data_config=data_config)
            if expected == seq_len:
                add('bin-grid consistency', 'PASS',
                    f'seq_len={seq_len} == get_total_steps(data_config)')
            else:
                add('bin-grid consistency', 'FAIL',
                    f'seq_len={seq_len} != get_total_steps={expected}')
    except Exception as exc:
        add('bin-grid consistency', 'FAIL', f'{type(exc).__name__}: {exc}')

    # --- 4. Synthetic forward pass --------------------------------------------
    ctx = None
    try:
        example_path = handoff / 'examples' / 'synthetic_patient.json'
        if example_path.is_file():
            raw_data = json.loads(example_path.read_text(encoding='utf-8'))
            source = 'examples/synthetic_patient.json'
        else:
            from astra.inference.synthetic import make_synthetic_raw_data
            raw_data = make_synthetic_raw_data(session.bundle)
            source = 'generated in-memory'

        from astra.inference.patient_context import PatientContext
        ctx = PatientContext.create(raw_data, session.bundle)
        result = session.predict_from_context(ctx)
        prob = float(result.probability)
        if math.isfinite(prob) and 0.0 <= prob <= 1.0:
            add('synthetic forward pass', 'PASS',
                f'P={prob:.4f} traj_len={ctx.trajectory_length} ({source})')
        else:
            add('synthetic forward pass', 'FAIL',
                f'probability out of [0, 1]: {prob!r}')

        if session.is_temporal:
            curve = result.predictions_over_time
            if curve is not None and len(curve) == seq_len:
                add('temporal prediction curve', 'PASS',
                    f'predictions_over_time length == seq_len ({seq_len})')
            else:
                got = None if curve is None else len(curve)
                add('temporal prediction curve', 'FAIL',
                    f'expected length {seq_len}, got {got}')
    except Exception as exc:
        add('synthetic forward pass', 'FAIL', f'{type(exc).__name__}: {exc}')
        logger.debug('Forward pass failure', exc_info=True)

    # --- 5. Optional SHAP smoke test -------------------------------------------
    if explain_smoke:
        if ctx is None:
            add('SHAP explanation smoke', 'SKIP', 'no patient context (predict failed)')
        else:
            try:
                shap_result = session.explain_from_context(ctx)
                if shap_result.ts_shap and len(shap_result.ts_shap) > 0:
                    add('SHAP explanation smoke', 'PASS',
                        f'{len(shap_result.ts_shap)} TS channels attributed')
                else:
                    add('SHAP explanation smoke', 'FAIL', 'ts_shap is empty')
            except ImportError as exc:
                add('SHAP explanation smoke', 'SKIP', f'shap not installed ({exc})')
            except Exception as exc:
                add('SHAP explanation smoke', 'FAIL', f'{type(exc).__name__}: {exc}')
                logger.debug('SHAP smoke failure', exc_info=True)

    failed = any(status == 'FAIL' for _, status, _ in rows)
    _print_summary(rows)
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# SELF-TEST (export + validate on generated tiny artifacts)
# ---------------------------------------------------------------------------

def run_self_test(explain_smoke=False):
    """End-to-end smoke test with NO real artifacts.

    Generates tiny synthetic artifacts (``save_tiny_artifacts``), exports
    them, then validates the exported bundle. Lets anyone verify the tooling
    (and their Python environment) without model or data access.

    Returns:
        0 on PASS, 1 on FAIL.
    """
    import tempfile

    from astra.inference.synthetic import save_tiny_artifacts

    tmp = tempfile.mkdtemp(prefix='astra_export_selftest_')
    try:
        tmp_path = Path(tmp)
        logger.info('Self-test working directory: %s', tmp_path)
        print(f'Self-test: generating tiny artifacts in {tmp_path} ...')

        artifacts = save_tiny_artifacts(str(tmp_path / 'artifacts'),
                                        model_name='tinytest')
        support = write_minimal_support_files(tmp_path / 'support')

        run_export(
            artifacts['model_name'],
            artifacts_dir=artifacts['artifacts_dir'],
            out_dir=str(tmp_path / 'handoff'),
            config_path=support['config'],
            metadata_csv=support['metadata_csv'],
            handoff_doc=str(tmp_path / 'HANDOFF.md'),   # absent → skipped
        )
        rc = run_validate(str(tmp_path / 'handoff'),
                          model_name=artifacts['model_name'],
                          explain_smoke=explain_smoke)
        print(f"SELF-TEST {'PASSED' if rc == 0 else 'FAILED'}")
        return rc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _model_name_from_config(config_path):
    """Resolve ``model_name`` from a config YAML (config-first CLI pattern)."""
    import yaml
    try:
        with open(config_path, encoding='utf-8') as fh:
            cfg = yaml.safe_load(fh) or {}
    except OSError as exc:
        raise SystemExit(
            f'--model-name not given and config not readable: {config_path} ({exc})')
    model_name = cfg.get('model_name')
    if not model_name:
        raise SystemExit(
            f"--model-name not given and {config_path} has no 'model_name' key")
    logger.info('Resolved model_name=%r from %s', model_name, config_path)
    return str(model_name)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python -m astra.inference.export_artifacts',
        description='Export / validate ASTRA deployment artifact bundles.',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    exp = sub.add_parser(
        'export', help='Collect + fingerprint artifacts into a handoff directory.')
    exp.add_argument('--model-name', default=None,
                     help='Model name (default: model_name from --config).')
    exp.add_argument('--artifacts-dir', default='models',
                     help='Source artifacts directory (default: models).')
    exp.add_argument('--out', default='handoff',
                     help='Destination handoff directory (default: handoff).')
    exp.add_argument('--config', default='configs/defaults.yaml',
                     help='Config file to ship (default: configs/defaults.yaml).')
    exp.add_argument('--no-ebm', action='store_true',
                     help='Skip EBM model files even if the model uses _ebm_pred.')
    exp.add_argument('--dry-run', action='store_true',
                     help='Report what would be copied; write nothing.')
    exp.add_argument('--sign-off', default=None,
                     help='Record data-protection approval for the patient-'
                          'derived shap_background in manifest.json, e.g. '
                          '"Approved <name> <date>: receiving team holds '
                          'full data rights". Omitting it leaves sign_off '
                          'null and prints a warning.')

    val = sub.add_parser(
        'validate', help='Acceptance test: hash check + model load + forward pass.')
    val.add_argument('--dir', dest='handoff_dir', default=None,
                     help='Handoff directory to validate.')
    val.add_argument('--model-name', default=None,
                     help='Model name (default: read from manifest.json).')
    val.add_argument('--explain-smoke', action='store_true',
                     help='Also run a SHAP explanation smoke test.')
    val.add_argument('--self-test', action='store_true',
                     help='Generate tiny artifacts, export and validate them '
                          'end-to-end (no real artifacts needed).')

    args = parser.parse_args(argv)

    if args.command == 'export':
        try:
            model_name = args.model_name or _model_name_from_config(args.config)
            run_export(
                model_name,
                artifacts_dir=args.artifacts_dir,
                out_dir=args.out,
                config_path=args.config,
                include_ebm=not args.no_ebm,
                dry_run=args.dry_run,
                sign_off=args.sign_off,
            )
            return 0
        except Exception as exc:
            logger.error('Export failed: %s', exc)
            logger.debug('Export failure', exc_info=True)
            print(f'EXPORT FAILED: {exc}')
            return 1

    # validate
    if args.self_test:
        return run_self_test(explain_smoke=args.explain_smoke)
    if not args.handoff_dir:
        parser.error('validate requires --dir HANDOFF_DIR (or --self-test)')
    try:
        return run_validate(args.handoff_dir,
                            model_name=args.model_name,
                            explain_smoke=args.explain_smoke)
    except Exception as exc:
        logger.error('Validation crashed: %s', exc)
        logger.debug('Validation failure', exc_info=True)
        print(f'VALIDATION FAILED: {exc}')
        return 1


if __name__ == '__main__':
    from astra.utils import setup_logging
    setup_logging()
    sys.exit(main())
