"""Diagnose a shap_cache.pkl <-> config channel-count mismatch. Read-only.

Reports three independent facts and tells you which artifact is stale:
  1. channel count baked into the cached SHAP results
  2. channel count implied by the current config's data dict
  3. channel count the trained checkpoint actually expects (W_P input width)
plus mtimes, so you can see the chronological order of the artifacts.

Usage:
    python -m scripts.diagnose_shap_cache [--config defaults.yaml]
"""
import argparse
import datetime as _dt
import os
import pickle

import numpy as np


def _mtime(path):
    if not os.path.exists(path):
        return "MISSING"
    ts = os.path.getmtime(path)
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="defaults.yaml")
    args = ap.parse_args()

    import astra.utils as _utils
    from astra.utils import get_cfg
    _cfg = get_cfg(_utils.PROJECT_ROOT / "configs" / args.config)
    _utils.cfg.clear()
    _utils.cfg.update(_cfg)
    from astra.utils import cfg

    model_name = cfg["model_name"]
    cache_path = "reports/shap_paper/shap_cache.pkl"

    print("=" * 78)
    print(f"config={args.config}   model_name={model_name!r}")
    print("=" * 78)

    # ---- 1. cached SHAP -------------------------------------------------
    print("\n[1] CACHED SHAP RESULTS")
    cached_c = None
    if not os.path.exists(cache_path):
        print(f"    {cache_path}: MISSING")
    else:
        with open(cache_path, "rb") as f:
            res = pickle.load(f)
        print(f"    {cache_path}  (mtime {_mtime(cache_path)})")
        print(f"    timepoint labels: {list(res.keys())}")
        for label, r in res.items():
            if not isinstance(r, dict):
                continue
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    print(f"      {label:>5} / {k:<22} shape={v.shape}")
            ts = r.get("ts_shap")
            if ts is not None and ts.ndim >= 2:
                cached_c = ts.shape[1]
            break  # one label is enough to read the channel count
        print(f"    -> cached ts_shap channel count: {cached_c}")

    # ---- 2. current config's data dict ----------------------------------
    print("\n[2] CURRENT CONFIG DATA DICT")
    from astra.data.caching import prepare_data_and_dls_cached
    from astra.evaluation.behavior import (
        create_channel_mapping, _get_clinical_only_channel_mask,
    )
    data = prepare_data_and_dls_cached(cfg)
    c_in = data["c_in"]
    channel2feature, _ = create_channel_mapping(data)
    clinical = _get_clinical_only_channel_mask(channel2feature, c_in)
    print(f"    data['c_in'] = {c_in}")
    print(f"    clinical indices ({len(clinical)}): {clinical}")
    print(f"    max clinical index = {max(clinical) if clinical else 'n/a'}")
    print("    full channel map:")
    for i in range(c_in):
        tag = "clinical" if i in clinical else "EXCLUDED"
        print(f"      [{i:>2}] {tag:<9} {channel2feature.get(i, '<unmapped>')}")

    # ---- 3. what the checkpoint expects ---------------------------------
    print("\n[3] TRAINED CHECKPOINT")
    ckpt = f"models/{model_name}.pth"
    print(f"    {ckpt}  (mtime {_mtime(ckpt)})")
    if os.path.exists(ckpt):
        import torch
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        for k, v in sd.items():
            if "W_P" in k and hasattr(v, "shape"):
                print(f"      {k}: {tuple(v.shape)}")
        excl = data.get("exclude_channel_indices", [])
        print(f"    data['exclude_channel_indices'] = {excl}")
        print(f"    -> W_P input width should equal c_in - len(excluded) = "
              f"{c_in} - {len(excl)} = {c_in - len(excl)}")

    # ---- 4. chronology ---------------------------------------------------
    print("\n[4] ARTIFACT CHRONOLOGY (oldest first tells you what is stale)")
    paths = [
        cache_path,
        "reports/shap_paper/stratified_samples.pkl",
        ckpt,
        f"reports/eval/{model_name}/predictions/time_metrics_{model_name}.csv",
        f"reports/eval/{model_name}/predictions/preds_df_{model_name}.csv",
        f"models/calibrators/{model_name}",
    ]
    rows = [(p, _mtime(p)) for p in paths]
    for p, m in sorted(rows, key=lambda r: (r[1] == "MISSING", r[1])):
        print(f"    {m}   {p}")

    # ---- verdict ---------------------------------------------------------
    print("\n" + "=" * 78)
    if cached_c is None:
        print("VERDICT: no cached SHAP to compare.")
    elif cached_c == c_in:
        print(f"VERDICT: channel counts MATCH ({cached_c}). The IndexError has another cause.")
    else:
        print(f"VERDICT: MISMATCH - cached SHAP has {cached_c} channels, "
              f"current config produces {c_in}.")
        print("  The checkpoint loading successfully under this config is strong evidence")
        print("  the CONFIG is right and shap_cache.pkl predates a channel change.")
        print("  Compare the mtimes above before deciding to --recompute.")
    print("=" * 78)


if __name__ == "__main__":
    main()
