# ASTRA Training Pipeline

## CLI Usage

```bash
# Full pipeline: pretrain → finetune on full trainval → eval
python -m astra.training.train --pretrain --finetune --eval

# Full pipeline with HP sweep: pretrain → HPO → retrain on full trainval → eval
python -m astra.training.train --pretrain --sweep-train --finetune --eval

# Finetune only (existing pretrained checkpoint, full trainval)
python -m astra.training.train --finetune --eval

# Finetune with 80/20 validation split + early stopping
python -m astra.training.train --finetune --no-skip-valid --eval

# Architecture sweep (Stage 1)
python -m astra.training.train --sweep-arch --n-arch-trials 30

# Training HP sweep only (no final retrain)
python -m astra.training.train --sweep-train --no-finetune --n-train-trials 50

# Comprehensive time-dependent evaluation
python -m astra.training.train --no-finetune --eval --comprehensive-eval --multicurve
```

## Pipeline Overview

```mermaid
flowchart TD
    CLI["python -m astra.training.train"] --> LOAD

    subgraph DATA["Data Loading"]
        LOAD["prepare_data_and_dls_cached(cfg)"]
        CACHE{"Cache\nhit?"}
        LOAD --> CACHE
        CACHE -- hit --> READY["data dict ready"]
        CACHE -- miss --> PREP
        PREP["prepare_data_and_dls(cfg)"] --> READY
    end

    READY --> P0 & S1 & S2 & FT_BLOCK

    P0{"--pretrain?"}
    P0 -- yes --> PRETRAIN["MLM Pretraining"]
    P0 -- no --> SKIP_PT[" "]:::hidden

    S1{"--sweep-arch?"}
    S1 -- yes --> ARCH["Architecture Sweep\n(Stage 1)"]
    S1 -- no --> SKIP_S1[" "]:::hidden

    S2{"--sweep-train?"}
    S2 -- yes --> SWEEP["Training HP Sweep\n(Stage 2)"]
    S2 -- no --> SKIP_S2[" "]:::hidden

    SWEEP --> RETRAIN_Q{"--finetune\n& --skip-valid?"}
    RETRAIN_Q -- yes --> RETRAIN["Retrain on full trainval\nwith best trial's\nactual epoch counts"]
    RETRAIN_Q -- no --> BEST_CFG["best_finetune_cfg"]

    RETRAIN --> EVAL_Q
    BEST_CFG --> FT_BLOCK

    FT_BLOCK{"--finetune\n& not sweep_retrained?"}
    FT_BLOCK -- yes --> FINETUNE["4-Phase Finetuning"]
    FT_BLOCK -- no --> EVAL_Q

    FINETUNE --> EVAL_Q
    PRETRAIN -.-> FINETUNE
    ARCH -.-> PRETRAIN

    EVAL_Q{"--eval?"}
    EVAL_Q -- yes --> EVAL["Evaluation\n(holdout set)"]
    EVAL_Q -- no --> DONE["Done"]
    EVAL --> DONE

    classDef hidden fill:none,stroke:none;
```

## Data Pipeline

```mermaid
flowchart LR
    subgraph RAW["Raw Data"]
        BASE["base_df\n(demographics,\ntrajectories)"]
        CONCEPTS["Concept CSVs\n(vitals, labs,\nmeds, procedures)"]
    end

    subgraph SPLIT["Temporal Split"]
        TRAINVAL["trainval\n(before 2023-06-01)"]
        HOLDOUT["holdout\n(after 2023-06-01)"]
    end

    subgraph ENCODE["Encoding & Normalization"]
        DF2XY["df2xy_pure()\nX: [n, channels, seq_len]\nchannels sorted by FEATURE"]
        NORM["Per-channel StandardScaler\nfit on trainval only\nmeasured → ~N(0,1)\nmissing/padding → 0.0"]
        TAB["TabularEncoder\n+ StandardScaler\nfit on trainval"]
        MHOT["MultiHotCategoricalEncoder\ncategorical TS → multi-hot\n[n, n_categories, seq_len]"]
    end

    subgraph DL["Dataloaders"]
        TV_DL["trainval\nAstraMixedDataLoader\n(splits=None, full dataset)"]
        HO_DL["holdout\nAstraMixedDataLoader\n(apply fitted scalers)"]
    end

    RAW --> SPLIT
    TRAINVAL --> DF2XY --> NORM --> TAB --> MHOT --> TV_DL
    HOLDOUT --> HO_DL
```

## MLM Pretraining

```mermaid
flowchart TD
    subgraph PRETRAIN["run_pretrain()"]
        VALID["Validate normalization\n(mean≈0, std≈1)"]
        SPLIT["80/20 stratified split\n(unlabeled, targets unused)"]
        MODEL["get_backbone()\n→ TSTabFusionMLM(backbone, MLMConfig)"]
        OPT["AdamW lr=5e-5\nCosine warmup (3 epochs)\nEarlyStopping patience=3"]

        subgraph EPOCH["Per Epoch"]
            subgraph BATCH["Per Batch"]
                MASK["Mask 4 modalities independently (15% each):\n80% zero/mean replace\n10% random timestep\n10% keep original"]
                LOSS["5 reconstruction losses:\n1. Cont TS MSE (signal channels only)\n2. Cat TS BCE per-feature\n3. Static cat cross-entropy\n4. Static cont MSE\n5. NT-Xent contrastive\nWeights: ts=1.5, cat_ts=1.5,\ncat=1.0, cont=1.0, contrastive=0.5"]
            end
            VAL_LOSS["Validate → save best model\npretrain_checkpoints/{name}/best_model.pt"]
        end

        VALID --> SPLIT --> MODEL --> OPT --> EPOCH
        MASK --> LOSS
        EPOCH --> VAL_LOSS
    end
```

## 4-Phase Finetuning

```mermaid
flowchart TD
    subgraph FINETUNE["run_finetune_v2()"]
        BACKBONE{"use_pretrained?"}
        LOAD_PT["load_pretrained_backbone()\nLoad checkpoint → extract backbone\nExpand W_P if c_in mismatch\nstrict=False if temporal_head"]
        FRESH["get_backbone()\nRandom init"]

        BACKBONE -- yes --> LOAD_PT --> DL_MODE
        BACKBONE -- no --> FRESH --> DL_MODE

        DL_MODE{"skip-valid?\n(valid_size ≤ 0)"}
        FULL["Full trainval\nvalid_dl = None\nNo early stopping"]
        SPLIT["80/20 stratified split\nEarlyStopping patience=7"]

        DL_MODE -- "default: yes" --> FULL
        DL_MODE -- "--no-skip-valid" --> SPLIT

        FULL --> P1
        SPLIT --> P1

        P1["Phase 1: Head-only\nfreeze all but classification head\n5 epochs, lr=1e-3"]
        P2["Phase 2: Partial unfreeze\nupper transformer layers + head\n12 epochs, lr=3e-4\ndiscriminative LRs (decay=0.1)"]
        P3["Phase 3: Full finetune\nall layers trainable\n8 epochs, lr=1e-4\ndiscriminative LRs"]
        P4_Q{"--early-prediction?"}
        P4["Phase 4: Early prediction hardening\n8 epochs, lr=5e-5\nprogressive time masking (prob=0.5)\nweighted loss (sparse data → higher weight)"]

        P1 --> P2 --> P3 --> P4_Q
        P4_Q -- yes --> P4 --> SAVE
        P4_Q -- no --> SAVE

        SAVE_Q{"Had validation?"}
        SAVE_Q -- yes --> RESTORE["EarlyStopping.restore_best()\nRevert to best checkpoint\nacross all phases"]
        SAVE_Q -- no --> FINAL["Use final model state"]

        SAVE --> SAVE_Q
        RESTORE --> MODEL_SAVE
        FINAL --> MODEL_SAVE

        MODEL_SAVE["save_model() → models/{name}.pth\nsave_deployment_bundle()\n→ models/deployment/deployment_{name}.pkl"]
    end
```

## HP Sweep → Retrain Flow

```mermaid
flowchart TD
    subgraph SWEEP["run_training_sweep()"]
        STUDY["Optuna: TPESampler + MedianPruner\nmaximize AUROC"]

        subgraph TRIAL["Per Trial (n=50)"]
            SUGGEST["Suggest HPs:\nfinetune_lr [1e-5..1e-3]\nweight_decay [1e-4..1e-1]\nlabel_smoothing [0..0.2]\nphase epochs, masking params"]
            RUN["run_finetune_v2()\n80/20 split, with Optuna trial\nReport AUROC per epoch → prune"]
            RECORD["Record actual epoch counts\n(accounting for early stopping)\nas trial.user_attrs"]
        end

        STUDY --> TRIAL
        SUGGEST --> RUN --> RECORD
        RECORD -- "next trial" --> SUGGEST
        RECORD -- "done" --> BEST["Best trial found"]

        BEST --> RETRAIN_Q{"retrain_full?"}
        RETRAIN_Q -- yes --> RETRAIN

        subgraph RETRAIN["Final Retrain"]
            READ["Read actual epoch counts\nfrom best_trial.user_attrs"]
            SET["valid_size = 0.0\n(full trainval, no early stopping)"]
            FINAL_RUN["run_finetune_v2()\nwith best HPs + actual epochs\non 100% trainval"]
        end

        READ --> SET --> FINAL_RUN
        RETRAIN_Q -- no --> CFG_ONLY["Return best_finetune_cfg\n(for manual retrain)"]
    end
```

## Evaluation

```mermaid
flowchart TD
    subgraph EVAL["run_eval()"]
        LOAD["prepare_model()\nLoad models/{name}.pth\nCUDA, eval mode"]

        LOAD --> TYPE{"temporal_head?"}

        subgraph STANDARD["Non-Temporal Path"]
            S_BASE["Full holdout forward pass\nROC/PR baseline"]
            S_MULTI{"--multicurve?"}
            S_MULTI_Y["8 key timepoints:\n1h, 6h, 12h, 24h, 72h, 7D, 13D, 30D\nCensor X at each step → forward pass\n→ ROC/PR curves per timepoint"]
            S_COMP{"--comprehensive?"}
            S_COMP_Y["~100 time steps\n(1h granularity 0-72h, 1D 4-30D)\nPer step: zero X beyond step,\nforward pass, AUROC + DeLong CI,\nAUPRC + bootstrap CI\n→ reports/predictions/\n→ reports/eval/"]
            S_BASE --> S_MULTI
            S_MULTI -- yes --> S_MULTI_Y --> S_COMP
            S_MULTI -- no --> S_COMP
            S_COMP -- yes --> S_COMP_Y
        end

        subgraph TEMPORAL["Temporal Path"]
            T_BASE["ONE forward pass\npreds: [n_patients, seq_len]\nBaseline: preds[:, last_valid_step]\nROC/PR plot"]
            T_MULTI{"--multicurve?"}
            T_MULTI_Y["Same 8 timepoints\nPick preds[:, min(step, traj_len-1)]\nNo extra forward passes"]
            T_COMP{"--comprehensive?"}
            T_COMP_Y["Same ~100 steps\nSlice existing preds\nNo extra forward passes\n→ same output format"]
            T_BASE --> T_MULTI
            T_MULTI -- yes --> T_MULTI_Y --> T_COMP
            T_MULTI -- no --> T_COMP
            T_COMP -- yes --> T_COMP_Y
        end

        TYPE -- "standard head" --> S_BASE
        TYPE -- "temporal head" --> T_BASE
    end
```

## Key Configuration (defaults.yaml)

| Parameter | Default | Section |
|---|---|---|
| Pretrain epochs | 3 | `pretrain.epochs` |
| Pretrain lr | 5e-5 | `pretrain.lr` |
| Phase 1 epochs / lr | 5 / 1e-3 | `finetune.phase1_*` |
| Phase 2 epochs / lr | 12 / 3e-4 | `finetune.phase2_*` |
| Phase 3 epochs / lr | 8 / 1e-4 | `finetune.phase3_*` |
| Phase 4 epochs / lr | 8 / 5e-5 | `finetune.phase4_*` |
| Validation size | 0.2 | `finetune.valid_size` |
| Early stopping patience | 7 | `finetune.patience` |
| LR decay factor | 0.1 | `finetune.lr_decay_factor` |
| Label smoothing | 0.1 | `finetune.label_smoothing` |
| Holdout split date | 2023-06-01 | `holdout_split_date` |
| Batch size | 64 | `training.bs` |
