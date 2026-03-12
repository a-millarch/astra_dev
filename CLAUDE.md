# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ASTRA (AI for Surgical Trauma Risk Assessment) is an ML-driven risk assessment tool for trauma patients, developed by CSTAR at Copenhagen University Hospital. It predicts 30-day mortality (`deceased_30d`) using a hybrid transformer model that fuses tabular, continuous time series, and categorical time series data from EHR records.

The dataset contains ~13,000 unique patients (~17,400 trainval samples). Input vectors are highly sparse: ~80% of timesteps are missing or padding. Continuous TS shape is `[batch, 27 channels, 114 timesteps]` with variable-width bins (10min→1h→1D).

Note: VSCode, where Claude Code functions, is separate from the development environment that contains the actual data. The user works in parallel in AzureML using git to push and pull code between secure and non-secure environment.

## Commands

### Environment Setup
```bash
conda create --name astra python=3.10.12 --no-default-packages -y
make requirements    # installs deps + editable package
make local           # installs package in editable mode only
```

### Data Processing
```bash
make data            # runs astra/make_data.py: base_df → bin_df → filter → map → TSDS
```

### Training & Evaluation (pure PyTorch, `python -m astra.training.train`)
```bash
make train_v2        # full pipeline: pretrain → 4-phase finetune → comprehensive eval
make finetune_v2     # finetune only (uses existing pretrained checkpoint)
make pretrain        # pretraining only (MLM self-supervised)
make eval            # evaluation only on existing models
make sweep           # two-stage HP sweep (architecture + training)
make sweep_arch      # architecture sweep only (30 trials)
make sweep_train     # training HP sweep only (50 trials)
```

### Direct CLI
```bash
# Full pipeline: pretrain → finetune on full trainval → eval
python -m astra.training.train --pretrain --finetune --eval

# Finetune only (existing checkpoint, full trainval, no validation split)
python -m astra.training.train --finetune --eval

# Finetune with 80/20 validation split + early stopping
python -m astra.training.train --finetune --no-skip-valid --eval

# Finetune with early prediction hardening (Phase 4)
python -m astra.training.train --finetune --early-prediction --eval

# HP sweep: architecture then training
python -m astra.training.train --sweep-arch --sweep-train --eval
```

Key flags: `--pretrain`, `--finetune/--no-finetune`, `--eval/--no-eval`, `--use-pretrained/--no-use-pretrained`, `--skip-valid/--no-skip-valid` (default: skip), `--early-prediction`, `--comprehensive-eval`, `--multicurve`, `--sweep-arch`, `--sweep-train`, `--validate-temporal`

Note: `--skip-valid` (default) trains on full trainval without validation. `--no-skip-valid` creates an 80/20 split with early stopping.

Legacy CLI (`astra/models/hybrid/train_model.py`) still works for basic pretrain/finetune/eval but lacks sweep and early-prediction support.

### Documentation
```bash
make build_documentation   # build MkDocs
make serve_documentation   # serve locally
```

## Architecture

### Configuration
All config lives in `configs/defaults.yaml`, loaded globally via `astra.utils.get_cfg()` → accessible as `cfg` dict. Key sections:
- `dataset`: exclusions, concepts, bin intervals, aggregation functions, categorical/numeric columns
- `model`: architecture HPs (`d_model`, `n_layers`, `n_heads`, `fc_dropout`, `res_dropout`, `temporal_head`, `causal`)
- `pretrain`: MLM masking probabilities, loss weights, contrastive learning, epochs/LR
- `finetune`: 4-phase training HPs (per-phase epochs/LR), regularization, validation, early stopping
- `sweep`: Optuna HP search space (architecture + training)
- `ebm_feature`: optional EBM predictions as input channel (`enabled: false` by default)
- `temporal_features`: positional encoding mode (`sinusoidal` from elapsed_hours)
- `evaluation`: threshold mode, F-beta settings

### Data Pipeline (`astra/data/`)

The pipeline processes clinical EHR data through these stages:

1. **`build_patient_info.py`** — Builds `base_df` (patient demographics, trajectories, comorbidities) and `bin_df` (time-binned structure).
2. **`filters.py`** — `filter_subsets_inhospital()` filters raw concept CSVs to in-hospital-only data, saves to `data/interim/concepts/`.
3. **`mapper.py`** — Maps clinical concepts to time bins with aggregation (min/max/mean/std/count). Output: `data/interim/mapped/`.
4. **`datasets.py`** — `TSDS` class merges mapped concepts into a unified dataset. `AggregatedDS` handles vectorized aggregation with optional GPU (cuDF/CuPy).
5. **`preprocessing.py`** — `MultiHotCategoricalEncoder` converts categorical TS (Medicin, Procedurer, ADTHaendelser) to count-based multi-hot tensors `[n_samples, n_categories, seq_len]`.
6. **`dataloader.py`** — `prepare_data_and_dls()` creates data dict with per-channel StandardScaler normalization (padding-aware via trajectory lengths). Missing measurements (NaN within trajectory) and padding (beyond trajectory) both become 0.0 after normalization; measured values become ~N(0,1).
7. **`mixed_dataloader.py`** — `AstraMixedDataLoader` wraps `AstraMixedDataset` into train/valid PyTorch DataLoaders. Also provides `df2xy_pure()`, `TabularEncoder`, model save/load utilities.
8. **`caching.py`** — `prepare_data_and_dls_cached()` caches the processed data dict for reproducibility.

Data flow: `data/raw/` → `data/interim/` (base_df, bin_df, concepts, mapped) → `prepare_data_and_dls()` → DataLoaders

Batch format: `((x_ts, (x_cat, x_cont), x_ts_cat), y)` where `x_ts` shape is `[batch, c_in, seq_len]`.

### Model (`astra/models/hybrid/`)

- **`model.py`** — `TSTabFusionTransformerMultiHot`: hybrid transformer fusing continuous TS, categorical multi-hot TS, and static tabular features via cross-attention (`_TabFusionEncoder`). Features:
  - `TimeAwarePositionalEncoding`: sinusoidal PE from elapsed_hours for temporal tokens, learned PE for statics
  - Optional `TemporalPredictionHead`: per-timestep predictions `[batch, seq_len]` with causal masking (each timestep uses only past information; statics are read-only context that cannot attend to temporal positions)
  - Auxiliary channels (elapsed_hours, `_data_present`) excluded from input projection `W_P` via `exclude_channel_indices`
  - Optional EBM channel: `_ebm_pred` from explainable boosting machine, projected through `W_P` like other clinical channels
- **`mlm.py`** — `TSTabFusionMLM` + `MLMConfig`: self-supervised pretraining with multi-task masked reconstruction (continuous TS, categorical TS, static categorical, static continuous) plus contrastive learning.
- **`training.py`** — `get_backbone()` factory + `run_pretrain()` orchestration.

### Training (`astra/training/`)

- **`train.py`** — CLI entry point (`python -m astra.training.train`). Orchestrates pretrain → sweep → finetune → eval.
- **`finetune.py`** — `FinetuneConfig` dataclass + `run_finetune_v2()`: 4-phase transfer learning:
  - Phase 1: Head-only training (backbone frozen, high LR to warm up head)
  - Phase 2: Partial unfreeze (upper transformer layers, discriminative LRs)
  - Phase 3: Full finetune (all layers unfrozen, lowest LR, discriminative across layer groups)
  - Phase 4: Optional early prediction hardening (progressive time masking + weighted loss for sparse samples)
  - Early stopping resets patience between phases but preserves globally best model state
- **`param_groups.py`** — Layer group extraction (embeddings → transformer pairs → head), discriminative LR computation, selective freezing (`freeze_to()`, `unfreeze_from()`, `unfreeze_all()`).
- **`sweep.py`** — Two-stage Optuna HP search: Stage 1 architecture search (d_model, n_layers, n_heads, dropout) + Stage 2 training HP search (LR, weight_decay, label_smoothing, phase epochs).
- **`scheduler.py`** — Cosine warmup LR scheduler (linear warmup then cosine annealing).
- **`utils.py`** — `EarlyStopping`, `MetricTracker`, `compute_auroc()`, checkpoint management.

Pretrained weights saved to `pretrain_checkpoints/{model_name}/`, finetuned models to `models/`.

### Evaluation (`astra/evaluation/`)

- **`predictive_performance.py`** — `TimeDependentEvaluator`: AUROC/AUPRC with CIs at multiple censoring time points. `run_eval()` is the entry point.
- **`calibration.py`** — ECE, isotonic regression, reliability diagrams.
- **`behavior.py`** — SHAP explainability, feature importance, channel mapping utilities.
- **`utils.py`** — `prepare_model()` loads trained backbone + device management. Time↔step conversion utilities.
- **`validate_temporal.py`** — Cross-validates `TemporalEvaluator` (single forward pass + causal mask) vs censored-dataloader evaluation.

### Inference (`astra/inference/`)

Single-patient real-time inference: `pipeline.py` orchestrates data preparation (`data_prep.py`), comorbidity scoring (`comorbidity.py`), optional EBM features (`ebm.py`), and patient context assembly (`patient_context.py`). Entry: `run_inference.py`.

### Key Concepts

- **Bin intervals**: Time discretized into variable-width bins (10min for 0-6h, 20min for 6-12h, 1h for 12-24h, up to 1D bins). Current config yields 114 timesteps per patient.
- **Concepts**: Clinical data sources (VitaleVaerdier, Labsvar, Medicin, Procedurer, ITAOversigtsrapport, ADTHaendelser), each with their own aggregation functions.
- **Temporal split**: Train/test split at `2023-06-01` (configurable). Patients before → trainval, after → holdout.
- **4-phase finetuning**: Head-only → partial unfreeze → full finetune → optional early prediction hardening. Discriminative LRs decay exponentially from head to embeddings.
- **Temporal head**: Optional per-timestep prediction mode (`temporal_head: true`). Uses causal masking so each position only attends to past timesteps. Statics are read-only context (blocked from attending to temporal positions to prevent information bridge).
- **EBM input feature**: Optional `_ebm_pred` channel from explainable boosting machine, injected into continuous TS as an additional feature. Enabled via `ebm_feature.enabled: true` in config.
- **Two operational modes**: Historic cohort mode (batch training) and single-patient continuous update mode (real-time inference).

### Dependencies

Core stack: PyTorch, pandas, scikit-learn, SHAP, Optuna. No FastAI or TSAI dependencies (pure PyTorch). Optional GPU acceleration via cuDF/CuPy. Azure ML integration for remote compute.

### Utilities (`astra/utils.py`)

`ProjectManager` manages project directories and logging (RichHandler + rotating file). Global `logger` and `cfg` are imported throughout the codebase.
