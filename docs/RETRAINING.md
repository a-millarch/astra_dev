# ASTRA retraining lifecycle — from a new data dump to redeployment

Companion to [HANDOFF.md](HANDOFF.md) (inference integration). This document covers the **full
cycle the operating team runs when a new raw data dump arrives**: rebuild the datasets, retrain
the model, evaluate it, and redeploy the artifact bundle for inference.

Background reading in this folder: [training_pipeline.md](training_pipeline.md) (training
architecture in depth), [inference_pipeline_diagram.md](inference_pipeline_diagram.md)
(deployment-bundle mechanics), [medication_profiles.md](medication_profiles.md) (feature
engineering rationale), [state.md](state.md) (pipeline state diagram).

```
new raw dump ──▶ 1. data pipeline ──▶ 2. train (pretrain→finetune) ──▶ 3. evaluate + calibrate
                     make_data              astra.training.train            (same command)
                                                                                 │
        5. swap service artifacts  ◀── 4. export + validate + parity gate  ◀─────┘
           (ASTRA_ARTIFACTS_DIR)        export_artifacts / azure_handoff_check
```

Everything is CLI-driven from the repository root. No Azure ML SDK is required — the pipeline
runs on flat CSV files in `data/raw/` (the Azure collector is used only when raw files are
absent *and* `azureml`/`mltable` is installed).

---

## 0. Prerequisites

| Requirement | Notes |
|---|---|
| Conda env | `conda env create -f environment_cpu.yml` (CPU) or `environment_gpu.yml`. **Training is realistic on GPU only** — pretrain + 4-phase finetune on ~17k samples takes hours on GPU and is impractically slow on CPU. Inference/data processing are CPU-fine. |
| `pip install -e .` | Package into the active env. |
| R + internet (optional) | `Rscript` on PATH with CRAN access, used ONCE per data rebuild for **one** batch feature: R-computed ISS (`icdpicr`; no Python equivalent). Degrades gracefully if unavailable: the `ISS_computed` channel stays empty (logged warning). The package self-installs into the gitignored `.r_libs/`. The Elixhauser score is computed in **Python** (same implementation as inference — `astra/inference/comorbidity.py`); set `ASTRA_ELIX_USE_R=1` to use the legacy R `comorbidity` package instead. |
| Disk layout | Working directory = repo root. `data/`, `models/`, `pretrain_checkpoints/`, `reports/`, `logging/` are created/used relative to it. |

## 1. Place the new dump

1. **Raw concept CSVs → `data/raw/`** — population-level files, one per concept
   (`VitaleVaerdier.csv`, `Labsvar.csv`, `Medicin.csv`, `Procedurer.csv`, `ADTHaendelser.csv`,
   `PatientInfo.csv`, `ITAOversigtsrapport.csv`, `EWS.csv`, `Notater.csv`, `Diagnoser.csv`, …).
   Same per-concept schema as the inference data contract (HANDOFF.md §4) — these are the
   *unfiltered* extracts covering **all** patients and the full time range.
2. **Cohort seed** — `data/raw/trauma_call_historic_population.csv`
   (columns `CPR_hash`, `ServiceDate`; one row per trauma call). If absent, it is **derived
   automatically** from `data/raw/Procedurer.csv`: every row with `ProcedureCode == 'BWST1F'`
   (the trauma-call procedure). Delete the old seed file when a new dump arrives so it is
   re-derived with the new patients.
3. **`data/external/metadata.csv`** — per-concept datetime-column definitions; stable across
   dumps, only changes if the extract format changes.
4. **Clear stale caches** from the previous dump:

```bash
rm -rf data/interim data/processed data/patients   # interim artifacts + per-patient inference cache
```

`data/patients/` matters: it is the inference-side per-patient CSV cache — serving it after a
new dump would silently pin patients to their *old* data.

## 2. Update the config

Edit **`configs/defaults.yaml`** directly — the batch pipeline (`make_data`, mappers,
dataloaders) reads it globally, so this file must reflect the retrain you are running.
(After training, keep a copy as `configs/<release>.yaml` if you want a frozen record; the
export/service steps accept `--config` / `ASTRA_CONFIG` pointing at either.) Keys to update:

| Key | Why |
|---|---|
| `model_name` | **New name per retrain** (e.g. a date stamp). All artifacts key off it: weights, deployment bundle, calibrators, reports. Never overwrite a deployed model's name. |
| `holdout_split_date` | Temporal train/holdout split (train ≤ date < holdout, `holdout_type: temporal`). Move it forward so the holdout stays a *recent, untouched* period — e.g. keep the last 6–12 months of the new dump as holdout. |
| `prehospital` | Keep `false` for phase-1 deployments (see HANDOFF.md §5). |
| `bin_intervals` / concepts / features | **Do not touch casually**: any change to the bin grid or channel set changes tensor shapes → existing pretrained checkpoints become incompatible and pretraining must run from scratch (it should anyway on a new dump — see below). |

## 3. Stage 1 — rebuild the datasets

```bash
python -m astra.make_data --overwrite        # base_df → bin_df → filter → map → cached data dict
make ebm_models                              # ONLY if ebm_feature.enabled: true (default false)
```

`--overwrite` regenerates `base_df`, `bin_df`, filtered concepts, mapped bins and the cached
data dict, but never touches `data/raw/`. Expect this to take on the order of **hours** for a
~13k-patient cohort (mapping dominates). Sanity checks while it runs:

- Cohort size logged after base_df creation — compare to the previous dump (should grow).
- Degradations log clearly: `skipping R-computed ISS` (no R), `ASMT_ELIX ... 0.0` (no
  `Diagnoser.csv`).
- On completion, the dataloader warns if the cached `seq_len` disagrees with
  `get_total_steps()` — that warning means a stale cache, rerun with `--overwrite`.

## 4. Stage 2+3 — retrain, evaluate, calibrate

```bash
python -m astra.training.train \
    --pretrain --finetune --eval --comprehensive-eval --calibrate
```

(Reads `configs/defaults.yaml`; `--config <yaml>` overrides for training-side settings, but
keep it consistent with what stage 1 was built with — the data cache is config-derived.)

- `--pretrain` — MLM self-supervised pretraining. **Re-pretrain on every new dump** (new data
  distribution; and mandatory if the grid/channels changed). Checkpoints →
  `pretrain_checkpoints/<model_name>/`.
- `--finetune` — 4-phase transfer learning on full trainval (default `--skip-valid`). On
  completion it **automatically writes the deployment bundle** →
  `models/deployment/deployment_<model_name>.pkl` (includes scalers, encoders, bin config and
  the SHAP background) and the weights → `models/<model_name>.pth`.
- `--eval --comprehensive-eval` — time-dependent AUROC/AUPRC with CIs on the temporal holdout →
  `reports/`. Optional extras: `--trauma-scores --delong` (baseline comparison), `--shap`
  (cohort SHAP), `--validate-temporal` (cross-validates the temporal-head evaluation).
- `--calibrate` — posthoc calibration (isotonic/Platt per timepoint) →
  `models/calibrators/<model_name>/` (picked up automatically at inference).

**Acceptance:** compare holdout AUROC/AUPRC and calibration (ECE) at the clinically relevant
censor times against the currently deployed model's report before proceeding.

## 5. Stage 4 — package, verify, redeploy

```bash
# One-command verification: synthetic self-test, real-model export+validate round trip,
# golden-patient parity (facade vs dashboard path), CLI smoke. Must end RESULT: PASS.
python scripts/azure_handoff_check.py --config configs/defaults.yaml \
    --sign-off "<who approved, when, basis>"

# Produce the shippable bundle (the config file itself is included in the bundle)
python -m astra.inference.export_artifacts export \
    --config configs/defaults.yaml --out handoff_<release>/ \
    --sign-off "<who approved, when, basis>"

# Receiving side / target machine: acceptance test (hashes → model load → synthetic forward)
python -m astra.inference.export_artifacts validate --dir handoff_<release>/ --explain-smoke
```

Then point the service at the new artifacts and restart:

```bash
export ASTRA_CONFIG=handoff_<release>/configs/defaults.yaml    # supplies model_name
export ASTRA_ARTIFACTS_DIR=handoff_<release>                   # bundle root (nested models/ resolved automatically)
python -m astra.service --port 8000
# GET /model/info must show the new model_name and expected seq_len/channels
```

**Rollback** is the same switch in reverse: keep the previous `handoff_<...>/` directory and
point `ASTRA_ARTIFACTS_DIR`/`ASTRA_CONFIG` back at it. Deployment bundles are self-contained —
nothing else to revert.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: Raw concept CSVs are missing ... collector unavailable` | The dump isn't in `data/raw/` (the Azure collector only exists on Azure ML). Place the CSVs. |
| `Population seed ... cannot be derived` | No seed file and no `data/raw/Procedurer.csv`. Provide either. |
| `skipping R-computed ISS` | R or CRAN unavailable (only affects the `ISS_computed` channel). Install R (`Rscript` on PATH) and rerun `make_data --overwrite` if wanted. |
| `ASMT_ELIX ... 0.0 for all patients` | `data/raw/Diagnoser.csv` missing or unreadable (Elixhauser itself is pure Python and needs no R). |
| `seq_len ... != get_total_steps()` warning | Stale cached data dict vs current bin config → rerun `python -m astra.make_data --overwrite`. |
| `load_pretrained_backbone` size-mismatch errors at finetune | Grid/channel change since the checkpoint → run with `--pretrain` (fresh pretraining). |
| Old values served for a patient after redeploy | Stale `data/patients/` per-patient cache from the previous dump — delete it (step 1.4). |
| Parity FAIL in `azure_handoff_check` | Investigate before deploying: the summary row names the failing stage and crash site; full traceback in `logging/astra.log`. |

Log files for every stage: `logging/astra.log` (rotating, DEBUG; `ASTRA_LOG_DIR` overrides).
