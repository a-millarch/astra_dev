"""
Synthetic tiny artifacts for the ASTRA inference tooling.

Builds a fully valid — but tiny and entirely fake — deployment bundle, model
and patient so that the export/validate tooling (and CI tests) can run
end-to-end WITHOUT access to real data or trained models.

Everything here mirrors the exact structures produced by
``astra.data.dataloader.save_deployment_bundle()`` and consumed by
``astra.inference.pipeline.InferenceSession.load()`` and
``astra.inference.patient_context.PatientContext.create()``:

- ``make_tiny_data_config()``  — minimal valid ``data_config`` (bin grid etc.)
- ``make_tiny_bundle()``       — full deployment-bundle dict with fitted scalers
- ``make_tiny_model()``        — randomly initialised model built from the bundle
- ``make_synthetic_raw_data()``— raw_data dict for ``PatientContext.create``
- ``save_tiny_artifacts()``    — writes bundle .pkl + weights .pth to disk

No real patient information is used anywhere in this module.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed, obviously-synthetic timestamps (far in the future).
ADMISSION_TIME = '2030-01-01 12:00:00'
CURRENT_TIME = '2030-01-01 14:00:00'   # 2h after admission

_DEFAULT_CHANNELS = ['HR', 'SBP', 'SPO2', 'TEMP']

# Plausible raw-scale (mean, std) per clinical feature. Used both to fit the
# tiny TS scaler and to sample synthetic measurements, so normalized inputs
# land in a sane ~N(0,1) range.
_FEATURE_STATS = {
    'HR': (85.0, 15.0),
    'SBP': (120.0, 20.0),
    'DBP': (70.0, 12.0),
    'MAP': (85.0, 12.0),
    'SPO2': (97.0, 2.0),
    'TEMP': (36.8, 0.5),
    'RESP': (16.0, 4.0),
    'GCS': (14.0, 1.5),
    'LACTATE': (1.8, 0.9),
}
_DEFAULT_STAT = (1.0, 0.5)

_MEDICATION_LABELS = ['ABX', 'ANALGESIA', 'VASOPRESSOR']

_DEFAULT_DEMOGRAPHICS = {
    'AGE': 54.0,
    'HEIGHT': 178.0,
    'WEIGHT': 82.0,
    'ASMT_ELIX': 3.0,
}


def _stats_for(feature):
    return _FEATURE_STATS.get(feature, _DEFAULT_STAT)


# ---------------------------------------------------------------------------
# Tiny data_config
# ---------------------------------------------------------------------------

def make_tiny_data_config(seq_len_target=None):
    """Build a small but VALID ``data_config`` dict.

    Same keys as ``save_deployment_bundle()`` stores under
    ``bundle['data_config']``. The default grid is
    ``{'1h': '10min', '3h': '30min', 'end': '1h'}`` → 6 + 4 = 10 steps
    (verified against ``get_total_steps``, the canonical seq_len source).

    Args:
        seq_len_target: Optional total step count. When given, the bin grid
            is adjusted so ``get_total_steps(data_config) == seq_len_target``.

    Returns:
        data_config dict with keys: bin_intervals, bin_freq_include,
        channel_map, ts_cat_names, cat_encoder_names, concepts,
        temporal_features, ebm_channel_idx, categorical_profiles.
    """
    from astra.evaluation.utils import get_total_steps

    if seq_len_target is None or seq_len_target == 10:
        bin_intervals = {'1h': '10min', '3h': '30min', 'end': '1h'}
        bin_freq_include = ['10min', '30min', '1h']
    elif seq_len_target >= 7:
        # 6 steps of 10min in the first hour, remainder as 30min bins.
        second_end_min = 60 + 30 * (seq_len_target - 6)
        bin_intervals = {'1h': '10min', f'{second_end_min}min': '30min', 'end': '1h'}
        bin_freq_include = ['10min', '30min', '1h']
    elif seq_len_target >= 1:
        bin_intervals = {f'{10 * seq_len_target}min': '10min', 'end': '1h'}
        bin_freq_include = ['10min', '1h']
    else:
        raise ValueError(f'seq_len_target must be >= 1, got {seq_len_target}')

    data_config = {
        'bin_intervals': bin_intervals,
        'bin_freq_include': bin_freq_include,
        'channel_map': {
            ch: {'concept': 'VitaleVaerdier', 'feature': ch,
                 'agg_func': 'mean', 'type': 'continuous'}
            for ch in _DEFAULT_CHANNELS
        },
        'ts_cat_names': [],
        'cat_encoder_names': {},
        'concepts': ['VitaleVaerdier'],
        'temporal_features': {},
        'ebm_channel_idx': None,
        'categorical_profiles': {},
    }

    total = get_total_steps(data_config=data_config)
    if seq_len_target is not None and total != seq_len_target:
        raise ValueError(
            f'Tiny bin grid yields {total} steps, expected {seq_len_target}. '
            f'bin_intervals={bin_intervals}'
        )
    logger.debug('Tiny data_config: %d steps, bin_intervals=%s', total, bin_intervals)
    return data_config


# ---------------------------------------------------------------------------
# Tiny deployment bundle
# ---------------------------------------------------------------------------

def make_tiny_bundle(n_channels=4, seq_len=None, temporal_head=False,
                     survival_mode=False, seed=0):
    """Build a complete deployment-bundle dict loadable by the SAME code paths
    as a real one (``InferenceSession.load`` / ``PatientContext.create``).

    Mirrors the key structure of ``save_deployment_bundle()`` exactly:
    ts_scaler, tab_scaler, encoding_info, cat_encoder, tab_feature_names,
    cat_feature_names, ts_channel_names, model_params, shap_background,
    data_config, ts_cat_profile_dims, profile_category_order, model_name.

    Notes:
        - The scalers are REAL fitted instances (``AstraScaler`` fitted via
          ``normalize_with_padding_mask``, sklearn ``StandardScaler`` for
          tabular) so ``normalize_new_patient()`` runs unmodified.
        - ``make_tiny_data_config()`` ships empty categorical defaults; this
          function upgrades ``ts_cat_names`` / ``cat_encoder_names`` so the
          bundle is self-consistent with ``ts_cat_dims`` / ``encoding_info``
          (one multi-hot 'medication' feature with 3 categories) and the
          categorical-TS code path is actually exercised.
        - ``survival_mode=True`` implies a temporal head; ``temporal_head=True``
          implies ``causal=True`` (matching training behaviour).

    Args:
        n_channels: Number of continuous TS channels (default channel names
            ``['HR', 'SBP', 'SPO2', 'TEMP']``, extended with ``CHi`` beyond 4).
        seq_len: Optional sequence length; grid is derived so that
            ``get_total_steps(data_config) == seq_len``.
        temporal_head: Build a per-timestep prediction model.
        survival_mode: Discrete-time survival head (forces temporal_head).
        seed: RNG seed for scaler-fit data and SHAP background.

    Returns:
        Bundle dict (in-memory; use :func:`save_tiny_artifacts` to persist).
    """
    from sklearn.preprocessing import StandardScaler

    from astra.data.dataloader import AstraScaler, normalize_with_padding_mask
    from astra.data.preprocessing import MultiHotCategoricalEncoder
    from astra.evaluation.utils import get_total_steps

    if survival_mode:
        temporal_head = True
    causal = bool(temporal_head)

    rng = np.random.default_rng(seed)

    channels = list(_DEFAULT_CHANNELS[:n_channels])
    while len(channels) < n_channels:
        channels.append(f'CH{len(channels)}')

    # --- data_config (bin grid + inference data-prep config) ---------------
    data_config = make_tiny_data_config(seq_len_target=seq_len)
    # Channel map restricted to this bundle's channels.
    data_config['channel_map'] = {
        ch: {'concept': 'VitaleVaerdier', 'feature': ch,
             'agg_func': 'mean', 'type': 'continuous'}
        for ch in channels
    }
    # Upgrade categorical config so it is consistent with ts_cat_dims below.
    data_config['ts_cat_names'] = ['Medicin']
    data_config['cat_encoder_names'] = {'Medicin': 'medication'}
    data_config['concepts'] = ['VitaleVaerdier', 'Medicin']

    actual_seq_len = get_total_steps(data_config=data_config)
    if seq_len is not None and actual_seq_len != seq_len:
        raise ValueError(
            f'seq_len mismatch: requested {seq_len}, grid yields {actual_seq_len}'
        )
    seq_len = actual_seq_len

    # --- categorical TS encoder (one feature, 3 categories) ----------------
    labels = list(_MEDICATION_LABELS)
    cat_encoder = MultiHotCategoricalEncoder()
    cat_encoder.encoders_['medication'] = {
        'value_to_idx': {v: i for i, v in enumerate(labels)},
        'idx_to_value': {i: v for i, v in enumerate(labels)},
        'n_classes': len(labels),
        'category_labels': labels,
    }
    cat_encoder.n_classes_['medication'] = len(labels)

    encoding_info = {
        'feature_ranges': {'medication': (0, len(labels))},
        'feature_names': ['medication'],
        'category_labels': {'medication': labels},
    }
    ts_cat_dims = {'medication': len(labels)}
    n_cat_dims = len(labels)

    # --- TS scaler: fit through the real normalization code path -----------
    n_fit = 32
    X_fit = np.full((n_fit, len(channels), seq_len), np.nan)
    for ch_idx, ch in enumerate(channels):
        mean, std = _stats_for(ch)
        vals = rng.normal(mean, std, size=(n_fit, seq_len))
        measured = rng.random((n_fit, seq_len)) < 0.7
        X_fit[:, ch_idx, :][measured] = vals[measured]
    low = max(1, seq_len // 2)
    fit_traj = rng.integers(low, seq_len + 1, size=n_fit)
    ts_scaler = AstraScaler(method='standard')
    normalize_with_padding_mask(X_fit, ts_scaler, fit_traj, fit=True)

    # --- Tabular scaler + static features -----------------------------------
    num_cols = ['AGE']
    ages = np.clip(rng.normal(55.0, 15.0, size=64), 18.0, 95.0)
    tab_scaler = StandardScaler().fit(pd.DataFrame({'AGE': ages}))

    classes = {'SEX': ['#na#', 'Female', 'Male']}  # TabularEncoder format
    cat_cols = ['SEX']

    # --- Model construction params (every key InferenceSession.load reads) --
    model_params = {
        'c_in': len(channels),
        'seq_len': seq_len,
        'classes': {k: list(v) for k, v in classes.items()},
        'cont_names': list(num_cols),
        'ts_cat_dims': dict(ts_cat_dims),
        'd_model': 16,
        'n_layers': 1,
        'n_heads': 2,
        'fc_dropout': 0.1,
        'res_dropout': 0.1,
        'fc_mults': (0.5, 0.25),
        'temporal_head': bool(temporal_head),
        'causal': causal,
        'temporal_head_dropout': 0.1,
        'temporal_head_mult': 0.5,
        'temporal_channel_idx': None,
        'bin_width_channel_idx': None,
        'exclude_channel_indices': [],
        'head_pool': 'flatten',
        'per_feature_cont_proj': False,
        'cat_ts_gate': False,
        'local_temporal_kernel': 1,
        'bin_width_modulation': False,
        'survival_mode': bool(survival_mode),
        'ts_cat_profile_dims': None,
    }

    # --- SHAP background (small random tensors, correct shapes/dtypes) ------
    bg_n = 8
    bg_ts = rng.normal(0.0, 1.0, size=(bg_n, len(channels), seq_len))
    bg_ts[rng.random(bg_ts.shape) < 0.5] = 0.0   # sparse like real normalized TS
    shap_background = {
        'ts': bg_ts.astype(np.float32),
        'ts_cat': (rng.random((bg_n, n_cat_dims, seq_len)) < 0.05).astype(np.float32),
        'cat': rng.integers(0, len(classes['SEX']), size=(bg_n, len(cat_cols))).astype(np.int64),
        'cont': rng.normal(0.0, 1.0, size=(bg_n, len(num_cols))).astype(np.float32),
    }

    bundle = {
        # --- Normalization artifacts ---
        'ts_scaler': ts_scaler,
        'tab_scaler': tab_scaler,
        'encoding_info': encoding_info,
        'cat_encoder': cat_encoder,
        'tab_feature_names': list(num_cols),
        'cat_feature_names': list(cat_cols),
        'ts_channel_names': list(channels),
        # --- Model construction params ---
        'model_params': model_params,
        # --- SHAP background data ---
        'shap_background': shap_background,
        # --- Data processing config (for inference data_prep) ---
        'data_config': data_config,
        # --- Profile-based categorical TS ---
        'ts_cat_profile_dims': None,
        'profile_category_order': None,
        # --- Metadata ---
        'model_name': 'tiny',
    }

    logger.info(
        'Tiny bundle built: c_in=%d seq_len=%d temporal_head=%s survival_mode=%s',
        len(channels), seq_len, temporal_head, survival_mode,
    )
    return bundle


# ---------------------------------------------------------------------------
# Tiny model
# ---------------------------------------------------------------------------

def make_tiny_model(bundle):
    """Construct a ``TSTabFusionTransformerMultiHot`` from
    ``bundle['model_params']`` exactly as ``InferenceSession.load`` does
    (c_out=2), with deterministic random init, in eval mode.
    """
    from astra.models.hybrid.model import TSTabFusionTransformerMultiHot

    params = bundle['model_params']
    torch.manual_seed(0)
    model = TSTabFusionTransformerMultiHot(
        c_in=params['c_in'],
        c_out=2,
        seq_len=params['seq_len'],
        classes=params['classes'],
        cont_names=params['cont_names'],
        ts_cat_dims=params['ts_cat_dims'],
        d_model=params['d_model'],
        n_layers=params['n_layers'],
        n_heads=params['n_heads'],
        fc_dropout=params['fc_dropout'],
        res_dropout=params['res_dropout'],
        fc_mults=params['fc_mults'],
        temporal_head=params['temporal_head'],
        causal=params['causal'],
        temporal_head_dropout=params['temporal_head_dropout'],
        temporal_head_mult=params.get('temporal_head_mult', 0.5),
        temporal_channel_idx=params.get('temporal_channel_idx', None),
        exclude_channel_indices=params.get('exclude_channel_indices', []),
        head_pool=params.get('head_pool', 'flatten'),
        per_feature_cont_proj=params.get('per_feature_cont_proj', False),
        cat_ts_gate=params.get('cat_ts_gate', False),
        local_temporal_kernel=params.get('local_temporal_kernel', 1),
        bin_width_channel_idx=params.get('bin_width_channel_idx', None),
        bin_width_modulation=params.get('bin_width_modulation', False),
        ts_cat_profile_dims=params.get('ts_cat_profile_dims', None),
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info('Tiny model built: %d parameters', n_params)
    return model


# ---------------------------------------------------------------------------
# Synthetic patient (raw_data for PatientContext.create)
# ---------------------------------------------------------------------------

def make_synthetic_raw_data(bundle, seed=0):
    """Generate a synthetic raw_data dict for ``PatientContext.create``.

    Follows the exact schema consumed by ``astra.inference.data_prep``:
    metadata keys (pid, admission_time, current_time, demographics) plus one
    entry per concept. Continuous concepts hold lists of
    ``{'timestamp', 'feature', 'value'}`` dicts; categorical concepts hold
    lists of ``{'timestamp', 'value'}`` point events. Timestamps are ISO
    strings so the result is directly JSON-serializable.

    Which concepts/features/categories are generated is driven entirely by
    the bundle (``data_config['channel_map']``, ``cat_encoder_names`` and
    ``encoding_info['category_labels']``), so this also works for REAL
    bundles, not just tiny ones.
    """
    rng = np.random.default_rng(seed)

    admission = pd.Timestamp(ADMISSION_TIME)
    current = pd.Timestamp(CURRENT_TIME)
    window_min = (current - admission).total_seconds() / 60.0

    data_config = bundle.get('data_config', {})
    channel_map = data_config.get('channel_map', {})
    ts_cat_names = set(data_config.get('ts_cat_names', []))
    cat_encoder_names = data_config.get('cat_encoder_names', {})
    encoding_info = bundle.get('encoding_info') or {}

    def _fmt(minutes):
        ts = admission + pd.Timedelta(minutes=float(minutes))
        return ts.strftime('%Y-%m-%d %H:%M:%S')

    # --- demographics -------------------------------------------------------
    demographics = {
        'AGE': 54.0,
        'SEX': None,
        'FIRST_HOSPITAL': None,
        'HEIGHT': 178.0,
        'WEIGHT': 82.0,
        'ASMT_ELIX': 3.0,
    }
    num_cols = list(bundle.get('tab_feature_names', []))
    means = getattr(bundle.get('tab_scaler'), 'mean_', None)
    for i, col in enumerate(num_cols):
        if means is not None and i < len(means) and np.isfinite(means[i]):
            demographics[col] = float(np.round(float(means[i]), 2))
        elif demographics.get(col) is None:
            demographics[col] = _DEFAULT_DEMOGRAPHICS.get(col, 50.0)
    classes = bundle.get('model_params', {}).get('classes', {})
    for col in bundle.get('cat_feature_names', []):
        class_list = list(classes.get(col, []))
        val = next((c for c in class_list if c != '#na#'), None)
        demographics[col] = str(val) if val is not None else 'unknown'

    raw_data = {
        'pid': 'SYNTH-0001',
        'admission_time': ADMISSION_TIME,
        'current_time': CURRENT_TIME,
        'demographics': demographics,
    }

    # --- continuous concepts (from channel_map) -----------------------------
    concept_features = {}
    for info in channel_map.values():
        if not isinstance(info, dict) or info.get('type') != 'continuous':
            continue
        concept = info.get('concept')
        if concept is None or concept in ts_cat_names or str(concept).startswith('_'):
            continue
        concept_features.setdefault(concept, set()).add(info['feature'])

    for concept, feats in sorted(concept_features.items()):
        events = []
        for feat in sorted(feats):
            mean, std = _stats_for(feat)
            times = np.sort(rng.uniform(2.0, max(window_min - 2.0, 3.0), size=5))
            for t in times:
                events.append({
                    'timestamp': _fmt(t),
                    'feature': str(feat),
                    'value': float(np.round(rng.normal(mean, std), 2)),
                })
        events.sort(key=lambda e: e['timestamp'])
        raw_data[concept] = events

    # --- categorical concepts (point events with known category labels) -----
    category_labels = encoding_info.get('category_labels', {})
    for concept, enc_name in sorted(cat_encoder_names.items()):
        labels = list(category_labels.get(enc_name, []))[:2]
        events = []
        for i, label in enumerate(labels):
            t = 15.0 + 30.0 * i
            if t >= window_min:
                t = window_min / 2.0
            events.append({'timestamp': _fmt(t), 'value': str(label)})
        raw_data[concept] = events

    n_cont = sum(len(v) for k, v in raw_data.items()
                 if k in concept_features)
    logger.debug(
        'Synthetic raw_data: %d continuous measurements across %d concepts, '
        '%d categorical concepts', n_cont, len(concept_features), len(cat_encoder_names),
    )
    return raw_data


# ---------------------------------------------------------------------------
# Persist tiny artifacts (bundle + weights) in the deployed layout
# ---------------------------------------------------------------------------

def save_tiny_artifacts(out_dir, model_name='tinytest', **kwargs):
    """Write tiny artifacts to *out_dir* in the same layout as production:

    - ``{out_dir}/deployment/deployment_{model_name}.pkl`` — pickled bundle
      (identical key structure to ``save_deployment_bundle``, readable by
      ``load_deployment_bundle``).
    - ``{out_dir}/{model_name}.pth`` — ``torch.save({'model': state_dict})``
      (the checkpoint format ``InferenceSession.load`` expects).

    Extra ``**kwargs`` are forwarded to :func:`make_tiny_bundle`
    (n_channels, seq_len, temporal_head, survival_mode, seed).

    Returns:
        Dict with keys: artifacts_dir, bundle_path, weights_path, model_name.
    """
    out_dir = Path(out_dir)
    bundle = make_tiny_bundle(**kwargs)
    bundle['model_name'] = model_name

    dep_dir = out_dir / 'deployment'
    dep_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = dep_dir / f'deployment_{model_name}.pkl'
    with open(bundle_path, 'wb') as f:
        pickle.dump(bundle, f)

    model = make_tiny_model(bundle)
    weights_path = out_dir / f'{model_name}.pth'
    torch.save({'model': model.state_dict()}, str(weights_path))

    logger.info('Tiny artifacts saved: %s, %s', bundle_path, weights_path)
    return {
        'artifacts_dir': str(out_dir),
        'bundle_path': str(bundle_path),
        'weights_path': str(weights_path),
        'model_name': model_name,
    }
