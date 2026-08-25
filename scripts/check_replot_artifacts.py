"""Preflight for replot_paper_figures: report which artifacts each config expects and
whether they exist. Read-only — touches nothing.

Usage:
    python -m scripts.check_replot_artifacts                     # all configs/*.yaml
    python -m scripts.check_replot_artifacts defaults.yaml       # one config
"""
import os
import sys

import astra.utils as _utils
from astra.utils import get_cfg

CONFIG_DIR = _utils.PROJECT_ROOT / "configs"
SKIP = {"profiles.yaml"}  # not a run config


def _mark(path, kind="file"):
    ok = os.path.isdir(path) if kind == "dir" else os.path.isfile(path)
    return f"  [{'x' if ok else ' '}] {path}"


def report(config_name):
    cfg = get_cfg(CONFIG_DIR / config_name)
    model_name = cfg["model_name"]
    print(f"\n{'=' * 72}\n{config_name}   ->   model_name = {model_name!r}\n{'=' * 72}")

    # Data cache key depends on data-affecting config params only
    try:
        _utils.cfg.clear()
        _utils.cfg.update(cfg)
        from astra.data.caching import _get_cache_key
        key = _get_cache_key(cfg)
        print(f"\ndata cache key: {key}")
        print(_mark(f"data/cache/data_cache_{key}.pkl"))
        print("      (missing -> falls back to full prepare_data_and_dls, slow but correct)")
    except Exception as e:
        print(f"\ndata cache key: FAILED to resolve ({type(e).__name__}: {e})")

    preds = f"reports/eval/{model_name}/predictions"
    print("\nrequired by replot_paper_figures:")
    print(_mark(f"{preds}/time_metrics_{model_name}.csv"))
    print(_mark(f"{preds}/time_metrics_{model_name}_active.csv"))
    print(_mark(f"{preds}/preds_df_{model_name}.csv"))
    print(_mark(f"{preds}/preds_df_{model_name}_active.csv"))

    print("\noptional (calibration figures skipped if absent):")
    print(_mark(f"models/calibrators/{model_name}", kind="dir"))

    print("\nrequired by shap_paper_figures --figures-only:")
    print(_mark(f"models/{model_name}.pth"))
    print("      (--figures-only still loads the model, even though it skips SHAP)")
    print(_mark("reports/shap_paper/shap_cache.pkl"))
    print(_mark("reports/shap_paper/stratified_samples.pkl"))
    print("      (stratified_samples.pkl is OVERWRITTEN on every run - back it up first)")


def main():
    names = sys.argv[1:]
    if not names:
        names = sorted(p.name for p in CONFIG_DIR.glob("*.yaml") if p.name not in SKIP)
    for n in names:
        try:
            report(n)
        except Exception as e:
            print(f"\n{n}: could not load ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
