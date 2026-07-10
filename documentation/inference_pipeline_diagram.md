# Inference Pipeline Architecture

## Overview

The inference pipeline enables single-patient prediction and SHAP explanation using a trained ASTRA model. It consists of two phases: saving deployment artifacts at training time, and loading them at inference time to run predictions without the full data pipeline.

## Flow Diagram

```mermaid
graph TD
    subgraph Training Time
        A[run_finetune / run_finetune_v2] --> B[save_model_fastai_compatible]
        A --> C[save_deployment_bundle]
        C --> D["models/deployment/deployment_{name}.pkl"]
        B --> E["models/{name}.pth"]
        C -->|extracts 200 samples| F[extract_shap_background]
        F -->|numpy arrays| D
        D -->|contains| D1[ts_scaler + tab_scaler]
        D -->|contains| D2[cat_encoder + encoding_info]
        D -->|contains| D3[model_params<br/>c_in, seq_len, classes,<br/>ts_cat_dims, d_model, ...]
        D -->|contains| D4[ts_channel_names]
        D -->|contains| D5[shap_background<br/>ts, ts_cat, cat, cont]
    end

    subgraph "InferenceSession.load()"
        E --> G[torch.load weights]
        D --> H[load_deployment_bundle]
        H --> I[Build model from<br/>saved model_params]
        G --> J[model.load_state_dict]
        I --> J
        J --> K[model.eval + to device]
        H --> L[Load SHAP background<br/>to GPU tensors]
        K --> M["InferenceSession"]
        L --> M
    end

    subgraph "predict(x_ts, x_ts_cat, tab_df)"
        N[Raw patient data] --> O["_prepare_tensors()"]
        O --> O1[Record NaN mask]
        O1 --> O2[Fill NaN with scaler mean]
        O2 --> O3["normalize_new_patient()<br/>ts: StandardScaler fit=False<br/>tab: StandardScaler fit=False"]
        O3 --> O4[get_trajectory_lengths]
        O3 --> O5[Encode static categoricals<br/>via saved classes dict]
        O5 --> O6[Handle _na indicators<br/>from NaN mask]
        O4 --> P[Tensors on device]
        O6 --> P
        P --> Q["model.forward()<br/>(x_ts, (x_cat, x_cont), x_ts_cat)"]
        Q -->|temporal head| R["sigmoid → [seq_len] probs<br/>pick prediction at censor_step"]
        Q -->|standard head| S["softmax → class 1 prob"]
        R --> T[InferenceResult]
        S --> T
    end

    subgraph "explain(x_ts, x_ts_cat, tab_df)"
        U[Same _prepare_tensors] --> V{censor_step?}
        V -->|yes| W[Zero out future data<br/>in patient + background]
        V -->|no| X[Use full data]
        W --> Y["_SHAPModelWrapper"]
        X --> Y
        Y -->|fixes| Y1[Passes causal_mask<br/>to transformer]
        Y -->|temporal| Y2["Returns logit at<br/>target_step only"]
        Y -->|standard| Y3["Returns [batch, 2]<br/>class logits"]
        Y1 --> Z[SHAP GradientExplainer<br/>background: 200 training samples]
        Y2 --> Z
        Y3 --> Z
        Z --> AA[Parse shap_values]
        AA -->|temporal| AB[Take output 0<br/>single target step]
        AA -->|standard| AC[Take class 1<br/>deceased]
        AB --> AD[Map to named features<br/>via ts_channel_names<br/>+ encoding_info]
        AC --> AD
        AD --> AE[SHAPResult<br/>ts_shap + cat_ts_shap +<br/>static_cat/cont_shap +<br/>top_features]
    end

    M --> N
    M --> U

    style D fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style M fill:#bbf,stroke:#333
    style T fill:#bfb,stroke:#333
    style AE fill:#bfb,stroke:#333
```

## Key Files

| File | Role |
|------|------|
| `astra/inference/pipeline.py` | `InferenceSession`, `_SHAPModelWrapper`, result dataclasses |
| `astra/data/dataloader.py` | `save_deployment_bundle()`, `load_deployment_bundle()`, `normalize_new_patient()` |
| `astra/training/finetune.py` | Calls `save_deployment_bundle()` after v2 finetuning |
| `astra/inference/export_artifacts.py` | Export/validate the deployment artifact bundle (`validate` doubles as the end-to-end smoke test) |
