# ASTRA Inference Handoff Guide

**Audience:** the engineering team implementing ASTRA inference in their own environment, against their
own EHR data feed (SQL, files, or message bus), with their own frontend rendering
probability-over-time and SHAP panels.

**What you receive:** an artifact bundle produced by `python -m astra.inference.export_artifacts export`
(section 2) plus this repository. You do **not** receive, and do not need, the training cohort data.

**What you build:** an adapter implementing the `PatientDataSource` protocol (section 4) and a frontend
that calls the Python API (section 6) or the bundled REST service (section 7).

**Operating the full lifecycle** (new data dump → rebuild datasets → retrain → evaluate →
redeploy) is covered separately in [RETRAINING.md](RETRAINING.md).

---

## 1. Overview & architecture

ASTRA predicts **30-day mortality** (`deceased_30d`) for trauma patients. The model
(`TSTabFusionTransformerMultiHot`, `astra/models/hybrid/model.py`) is a hybrid transformer fusing three
input streams built from raw EHR events:

1. **Continuous time series** `x_ts [n_channels, seq_len]` — vitals, labs, ICU scores, EWS score, ISS,
   binned onto a variable-width time grid (10-minute bins immediately after admission, widening to
   1-day bins later; the exact grid ships in the bundle as `data_config['bin_intervals']`).
2. **Categorical time series** `x_ts_cat [n_categories, seq_len]` — multi-hot medication categories,
   procedures, ward transfers (ADT), invasive-monitoring events, on the same grid.
3. **Static tabular features** — AGE, SEX, HEIGHT, WEIGHT, FIRST_HOSPITAL, Elixhauser comorbidity
   score (`ASMT_ELIX`), and optionally the prehospital ABCD assessment (A/B/C/D).

All of this is built for you by the inference stack. The single integration surface is:

```
your frontend ──HTTP──▶ astra.service (REST, optional) ──▶ AstraPredictor (astra/inference/api.py)
                                                              │
                                                              ├─ InferenceSession   (model + scalers + SHAP)
                                                              ├─ PatientContext     (per-patient tensors, binning)
                                                              └─ SimulationRunner   (non-temporal curve replay)
                                                              │
                                              load_patient_csv (astra/inference/patient_store.py)
                                                              │
                                            PatientDataSource.fetch(cpr_hash, concept)   ◀── YOUR ADAPTER
                                                              │
                                                       your EHR feed (SQL / files / memory)
```

`AstraPredictor` is the facade you program against:

| Method | Purpose |
|---|---|
| `AstraPredictor.load(model_name, artifacts_dir='models', *, device=None, data_source=None, data_dir='data/raw', patient_dir='data/patients', context_cache_size=8)` | Load model + bundle, register your data source |
| `predict(patient_id, timestamp, service_date, *, include_curve=True)` → `PredictionResponse` | Probability at `timestamp` + probability-over-time curve |
| `explain(patient_id, timestamp, service_date, *, top_n=20, include_values=True)` → `ExplanationResponse` | Full SHAP payload for all panels |
| `explain_differential(patient_id, service_date, t1_hours, t2_hours, *, include_endpoints=False)` → `DifferentialExplanationResponse` | ΔSHAP between two elapsed-hour timepoints (T2 − T1) |
| `explain_ebm(patient_id, timestamp, service_date)` → `Optional[dict]` | Local EBM contributions; `None` unless the model uses the `_ebm_pred` channel |
| `model_info()` → `dict` | Static metadata: channels, static features, bin grid, `TimeAxis`, library versions |
| `clear_cache(patient_id=None)` | Drop cached patient contexts |

Errors (defined in `astra/inference/api.py`) and the HTTP statuses the reference service maps them to:

| Exception | Meaning | HTTP |
|---|---|---|
| `PatientNotFoundError` | No data resolvable for the patient (data source returned nothing for required concepts) | 404 |
| `TimestampBeforeAdmissionError` | Requested timestamp precedes trajectory start | 422 |
| `ArtifactError` | Model artifacts missing/unloadable | 503 |
| `ValueError` | Unparseable timestamp / missing `service_date` | 422 |

### Temporal vs non-temporal models — curve cost

Both model types are handled transparently, but their cost profiles differ dramatically:

- **Temporal-head model** (`bundle['model_params']['temporal_head'] == True`, exposed as
  `is_temporal`): one forward pass yields per-timestep probabilities for the whole grid. The
  probability-over-time curve is essentially free (`curve.source == 'temporal_head'`).
- **Non-temporal model**: the curve is built by a `SimulationRunner`
  (`astra/inference/simulation.py`) stepping bin-by-bin through the grid — **one forward pass per
  visible bin** (order of 100 passes for a full trajectory; `curve.source == 'simulation'`).
  Contexts are LRU-cached so repeated/advancing queries only pay for new bins.

Check which you have via `model_info()['is_temporal']`.

---

## 2. Artifact bundle

The owner exports the bundle on the secure environment (config-first: the model
name is read from the config's `model_name` key, and that same config file ships
in the bundle; `--model-name` overrides it if needed):

```bash
python -m astra.inference.export_artifacts export --config configs/<experiment>.yaml --out handoff/
```

Contents (`<M>` = model name):

| Path in bundle | Contents | Required |
|---|---|---|
| `deployment/deployment_<M>.pkl` | Deployment bundle: `ts_scaler`, `tab_scaler`, `cat_encoder`, `encoding_info`, `ts_channel_names`, `tab_feature_names`, `cat_feature_names`, `model_params` (architecture + `classes` + `seq_len`), `shap_background`, `data_config` (bin grid, `channel_map`, `ts_cat_names`, `cat_encoder_names`, `concepts`) — see `save_deployment_bundle()` in `astra/data/dataloader.py` | yes |
| `<M>.pth` | Model weights (`{'model': state_dict, ...}` checkpoint) | yes |
| `calibrators/<M>/` | Post-hoc probability calibrators + `metadata.json` (`best_method`, `timepoints`) | no — without it, raw sigmoid/softmax probabilities are served |
| `ebm/` | Per-interval EBM deployment models | only if the model has an `_ebm_pred` channel |
| `configs/defaults.yaml` | Runtime config (read for SHAP settings and the `concepts` list) | yes |
| `data/external/metadata.csv` | Per-concept datetime column + time-window offset (section 4.2) | yes |
| `examples/synthetic_patient.json` | Fully synthetic patient in the raw-concept schema, for smoke tests | yes |
| `manifest.json` | `sha256` per file, library versions, model metadata, `shap_background` **sign-off field** | yes |

**Acceptance test.** After copying the bundle into your environment, run the validator — it verifies
hashes against `manifest.json`, loads the model, and runs a forward pass on the synthetic patient:

```bash
python -m astra.inference.export_artifacts validate --dir handoff/            # hash + load + forward
python -m astra.inference.export_artifacts validate --dir handoff/ --explain-smoke   # + SHAP smoke
```

A `PASS` from `validate` is the agreed acceptance criterion for "artifacts arrived intact and run here".

> ⚠️ **Data-protection callout.** `shap_background` inside `deployment_<M>.pkl` contains **~200
> patient-derived, normalized input tensors** from the training cohort (see
> `extract_shap_background()` in `astra/data/dataloader.py`). They are normalized and de-identified
> but derived from real patients. The `sign_off` field in `manifest.json` **must be completed by the
> data controller before the bundle leaves the secure environment** — record it at export time with
> `export --sign-off "<who approved, when, basis>"` (e.g. *"receiving team holds full data rights"*).
> Treat the bundle with the same confidentiality as pseudonymized patient data.

---

## 3. Install & run

```bash
git clone <repo>   # or unpack the source archive shipped with the bundle
cd astra

# The conda environment is the canonical dependency specification
# (CPU-only is sufficient; environment_gpu.yml exists for GPU machines)
conda env create -f environment_cpu.yml && conda activate astra_cpu

# Then install the package into it
pip install -e .
pip install -e .[service]        # extra: FastAPI/uvicorn for the REST service
```

Then place/point at the artifact bundle and either use the Python API directly (section 6) or run the
REST service (section 7).

**Runtime requirements:**

- **CPU is sufficient.** All deployment paths are CPU-tested; pass `device='cpu'` (or
  `ASTRA_DEVICE=cpu`) explicitly to pin it. GPU is auto-detected but not required.
- **Working directory must be the repository root.** The pipeline resolves relative paths at runtime:
  - `configs/defaults.yaml` (global config via `astra.utils.get_cfg()`),
  - `data/external/metadata.csv` (read inside `_filter_concepts_for_patient`, `astra/inference/data_prep.py`),
  - `data/interim/concepts/` — must be **writable**: `filter_vitals()` (`astra/data/filters.py`)
    persists the derived `InvasiveMonitoring.pkl` there, which `_filter_concepts_for_patient` reads back,
  - `data/patients/` — per-patient CSV cache (file-based mode only; writes are best-effort).
- One `AstraPredictor` per process. The registered data source is process-global
  (`set_data_source`), and a `threading.RLock` serializes inference — see section 9.
- **Log files.** Every entry point writes a rotating DEBUG log to `logging/astra.log`
  (daily rotation, 30-day retention) in addition to console output. Override the location
  with the `ASTRA_LOG_DIR` environment variable (set it empty to disable file logging);
  an unwritable directory degrades to console-only with a warning.

---

## 4. Data contract (normative)

This section defines exactly what your data feed must return. **The contract is a DataFrame schema
per concept, not a file format** (`astra/inference/datasource.py`). Everything downstream — EWS→vitals
merging, notes-derived features, Elixhauser scoring, trajectory building, binning, normalization —
runs unchanged on whatever your adapter returns.

### 4.1 The `PatientDataSource` protocol

```python
from typing import Optional, Protocol
import pandas as pd

class PatientDataSource(Protocol):                      # astra/inference/datasource.py
    def fetch(self, cpr_hash: str, concept: str) -> Optional[pd.DataFrame]:
        """Rows for one patient and one concept, or None if absent/not served."""

    # OPTIONAL capability — see section 4.14:
    # def fetch_prehospital(self, cpr_hash: str) -> Optional[dict]: ...
```

Skeleton SQL adapter (adapt queries/column aliases to your warehouse):

```python
import pandas as pd
from astra.inference.api import AstraPredictor

class SQLDataSource:
    """PatientDataSource backed by the hospital's SQL feed."""

    CONCEPT_QUERIES = {
        # Alias your columns to the exact names in sections 4.3–4.13:
        'PatientInfo':        'SELECT cpr AS "CPR_hash", dob AS "Fødselsdato", dod AS "Dødsdato", sex AS "Køn" FROM patient WHERE cpr = :cpr',
        'ADTHaendelser':      'SELECT cpr AS "CPR_hash", event AS "ADT_haendelse", t_in AS "Flyt_ind", t_out AS "Flyt_ud", dept AS "Afsnit" FROM adt WHERE cpr = :cpr',
        'VitaleVaerdier':     'SELECT cpr AS "CPR_hash", ts AS "Registreringstidspunkt", param AS "Vital_parametre", val AS "Værdi" FROM vitals WHERE cpr = :cpr',
        'Labsvar':            'SELECT cpr AS "CPR_hash", ts AS "Prøvetagningstidspunkt", test AS "BestOrd", val AS "Resultatværdi" FROM labs WHERE cpr = :cpr',
        # ... one query per concept you serve (Medicin, ITAOversigtsrapport, EWS,
        #     Notater, Diagnoser, Procedurer, ISS_computed)
    }

    def __init__(self, engine):
        self.engine = engine

    def fetch(self, cpr_hash: str, concept: str):
        query = self.CONCEPT_QUERIES.get(concept)
        if query is None:
            return None                                  # concept not served → pipeline degrades (section 5)
        df = pd.read_sql(query, self.engine, params={'cpr': cpr_hash})
        return df if len(df) else None                   # None/empty → treated as "concept absent"

predictor = AstraPredictor.load(
    "<MODEL>", artifacts_dir="handoff",
    data_source=SQLDataSource(engine),                   # registered process-globally
)
```

Behavioral notes (from `load_patient_csv` in `astra/inference/patient_store.py`):

- A registered source is **authoritative** — the file-system fallback is bypassed entirely.
- Returning `None` or an empty frame is interpreted as *"this concept is absent for this patient"*;
  callers degrade per section 5. Do not raise for missing data.
- If your frame contains a `CPR_hash` column it is defensively re-filtered to `cpr_hash`.
- `InMemoryDataSource` (`astra/inference/datasource.py`) is a ready-made in-memory implementation
  for tests: `InMemoryDataSource({cpr: {'VitaleVaerdier': df, ...}}, prehospital={cpr: {...}})`.
- `CSVDataSource` reads `{patient_dir}/{cpr_hash}/{concept}.csv` falling back to
  `{data_dir}/{concept}.csv` — the reference file-based adapter.

### 4.2 Universal conventions

| Rule | Detail |
|---|---|
| `CPR_hash` | `str`, stable pseudonymous patient ID, **identical across all concepts**. Every concept frame must carry it. |
| Timestamps | Naive local time (Europe/Copenhagen), **no timezone offsets**. Must be parseable by `pd.to_datetime` (`errors="coerce"`; ADT columns use `format="mixed"`). Recommended format: `YYYY-MM-DD HH:MM:SS`. Unparseable values become `NaT` and the rows are silently dropped by time filters. |
| Which column is "the" timestamp | Defined per concept in `data/external/metadata.csv` (`dt_colname`); table below. |
| Time-window filter | `filter_inhospital()` (`astra/data/filters.py`) keeps rows with `dt_colname` in `[trajectory_start − ts_offset days, trajectory_end + ts_offset days]`. `ts_offset` also comes from `metadata.csv`. |
| Decimal comma | Tolerated **only** in `Labsvar.Resultatværdi` (`filter_labs` and `_standardize_labs` replace `,`→`.`; `<`, `>`, `*` are stripped). All other numeric values must use `.` decimals or be integers. |
| Extra columns | Harmless — they survive filtering and are ignored. |
| Index artifacts | Return clean frames. (`index_col=0` handling applies only to the built-in CSV path.) |
| Missing concept | Return `None` — never fabricate empty placeholder rows. |

`data/external/metadata.csv` ships in the bundle and is read at runtime. Rows relevant to inference:

| `filename` (concept) | `dt_colname` (timestamp column) | `ts_offset` (days) |
|---|---|---|
| `ADTHaendelser` | `Flyt_ind` | 0 |
| `VitaleVaerdier` | `Registreringstidspunkt` | 0 |
| `Labsvar` | `Prøvetagningstidspunkt` | 0 |
| `Medicin` | `Administrationstidspunkt` | 0 |
| `Procedurer` | `ServiceDatetime` | 1 |
| `ITAOversigtsrapport` | `Målingstidspunkt` | 0 |
| `EWS` | `Målingstidspunkt` | 0 |
| `Notater` | `Redigeringstidspunkt` | 0 |
| `Diagnoser` | `Noteret_dato` | 1 |

The concept set a given model actually consumes is `bundle['data_config']['concepts']`
(also shown by `model_info()`); typical: `ITAOversigtsrapport`, `VitaleVaerdier`,
`InvasiveMonitoring` (derived), `Labsvar`, `Medicin`, `ADTHaendelser`, `EWS`, `ISS_notes` (derived),
`ISS_computed`. `PatientInfo`, `Notater` and `Diagnoser` are consumed even though they are not
"concepts" (identity, notes-derived features, comorbidity).

### 4.3 `PatientInfo` — required

Consumed by `_build_single_patient_base_df()` (`astra/inference/data_prep.py`). **Absent ⇒
`PatientNotFoundError`.** One row per patient is enough.

| Column | Type/format | Meaning | Required |
|---|---|---|---|
| `CPR_hash` | str | Patient ID | yes |
| `Fødselsdato` | datetime | Date of birth → `AGE` at admission | yes |
| `Dødsdato` | datetime or empty | Date of death (unused for the prediction itself) | column must exist; values optional |
| `Køn` | `'Mand'` / `'Kvinde'` | Sex → `SEX ∈ {Male, Female}` | yes |

### 4.4 `ADTHaendelser` (admission/discharge/transfer) — required

The backbone: trajectories, admission time, first hospital, and the ward-location categorical channel
are all derived from ADT. **Absent ⇒ `PatientNotFoundError`.**

| Column | Type/format | Meaning | Required |
|---|---|---|---|
| `CPR_hash` | str | Patient ID | yes |
| `ADT_haendelse` | str | Event type. Values with special semantics: `'Indlæggelse'` (admission — **starts a trajectory**; at least one required per admission), `'Flyt Ind'` (transfer-in — its `Flyt_ind` is nudged +1 s for ordering). Other values (e.g. discharge events) pass through. | yes |
| `Flyt_ind` | datetime | Interval start (this is the concept's `dt_colname`) | yes |
| `Flyt_ud` | datetime or empty | Interval end. Missing values are forward-filled from the next event's `Flyt_ind`, then from the trajectory end. | column yes; values optional |
| `Afsnit` | str | Department name, e.g. `'RH TRAUMECENTER'`, `'... INTENSIV ...'` | yes |

Processing (`build_trajectories`/`add_first_contacts` in `astra/data/build_patient_info.py`,
`filter_adt` in `astra/data/filters.py`):

- Consecutive admissions with gaps < 1 h are collapsed into one trajectory; the trajectory whose
  span contains `service_date` (± 1 day) is selected.
- `Afsnit` is classified into a location code via `classify_department()` using `ADT_PATTERNS`
  (`astra/data/mappings.py`): **`TB`** (traumecenter), **`ED`** (akutmodtagelse/akutklinik/modtagelse),
  **`OR`** (operations*/dagkirurgi/…), **`ICU`** (intensiv/ita), **`WARD`** (seng), **`OPD`** (amb).
  Unclassifiable departments are dropped from the categorical channel (but still count for
  trajectory/first-hospital logic).
- `Afsnit` also drives `FIRST_HOSPITAL` (static feature) via `derive_first_hospital()` →
  one of `VALID_HOSPITALS` (`RH`, `AHH`, `HGH`, `NOH`, `BFH`, `BOH`, `RHP`, `SJ KØGE`, …) or `MISC`.
  RH-prefixed departments additionally set the `LVL1TC` flag.

### 4.5 `VitaleVaerdier` (vital signs)

Filtered by `filter_vitals()` (`astra/data/filters.py`); also mined for static HEIGHT/WEIGHT by
`_extract_height_weight()` (`astra/inference/data_prep.py`).

| Column | Type/format | Meaning | Required |
|---|---|---|---|
| `CPR_hash` | str | Patient ID | yes |
| `Registreringstidspunkt` | datetime | Measurement time | yes |
| `Vital_parametre` | str | Parameter name (see below) | yes |
| `Værdi` | str/number | Value. Blood-pressure parameters carry `"sys/dia"` strings (e.g. `"120/80"`); everything else numeric with `.` decimals. `< / >` prefixes tolerated. | yes |
| `Værdi_Omregnet` | any | Present in the source extract; **not consumed** | no |

Parameter names actually consumed (`VITALS_MAP`, `BP_TYPES`, `HEIGHT_WEIGHT_MAP`,
`INVASIVE_VITALS_MAP`, `INVASIVE_BP_TYPES` in `astra/data/mappings.py`):

| Raw `Vital_parametre` value | → Feature | Notes |
|---|---|---|
| `Saturation` | `SPO2` | |
| `Puls`, `Puls (fra SAT-måler)`, `ABP Puls (fra A-kanyle)` | `HR` | `ABP Puls (fra A-kanyle)` also emits an invasive-monitoring event (`arterial_hr`) |
| `Resp.frekvens` | `RESPIRATORYRATE` | |
| `Temperatur`, `Temp.`, `Blæretemperatur`, `Esophagustemperatur` | `TEMP` | `Kernetemperatur`/`Blæretemperatur`/`Rektaltemperatur`/`Axiltemperatur`/`Esophagustemperatur` are converted °F→°C; `Temperatur` values > 50 are assumed °F and converted |
| `ART mean inv BT` | `MAP` | dropped by default config (`drop_features`) |
| `BT`, `NIBP`, `ART inv BT`, `ABP inv BT`, `Invasivt BT - ABP (sys/dia)`, `Invasivt BT - ART (sys/dia)` | `SBP` + `DBP` | split on `/`; the four invasive names also emit `arterial_bp` invasive events (`DBP` dropped by default config) |
| `Højde` | `HEIGHT` (static) | ⚠️ raw values are **inches** — converted via `inches_to_cm()` |
| `Vægt` | `WEIGHT` (static) | ⚠️ raw values are **ounces** — converted via `ounces_to_kg()` |

Physiological bounds (`VITALS_BOUNDS`) discard out-of-range values, e.g. SBP 0–300, HR 0–250,
SPO2 0–100, TEMP 25–43 °C, HEIGHT 50–230 cm, WEIGHT 2–300 kg. Unknown parameter names are ignored.
Continuous channels are named `{FEATURE}_{agg}` per the aggregations in the bundle's `channel_map`
(default: `mean` + `std` for vitals — e.g. `HR_mean`, `HR_std`).

### 4.6 `Labsvar` (lab results)

Filtered by `filter_labs()`.

| Column | Type/format | Meaning | Required |
|---|---|---|---|
| `CPR_hash` | str | Patient ID | yes |
| `Prøvetagningstidspunkt` | datetime | Sampling time | yes |
| `BestOrd` | str | Raw test name — **must match** a name in `LABS_FEATURE_MAP` (below); all other tests are discarded | yes |
| `Resultatværdi` | str | Numeric value as string; decimal **comma tolerated**; `*`, `<`, `>` stripped | yes |

Test names consumed (`LABS_FEATURE_MAP`, `astra/data/mappings.py`):

| Feature | Raw `BestOrd` names |
|---|---|
| `LACTATE` | `LAKTAT(POC);P(AB)`, `LAKTAT;P(AB)`, `LAKTAT;P(VB)`, `LAKTAT(POC);P(VB)`, `LAKTAT;CSV`, `LAKTAT(POC);CSV`, `LAKTAT(POC);P(KB)` |
| `BASE_EXCESS` | `BASE EXCESS;ECV`, `ECV-BASE EXCESS;(POC)` |
| `HEMOGLOBIN` | `HÆMOGLOBIN;B`, `HÆMOGLOBIN(POC);B`, `HÆMOGLOBIN (POC);B` |
| `LEUKOCYTES` | `LEUKOCYTTER;B` |
| `B-GROUP-LEUKOCYTES` | `LEUKOCYTTYPE (MIKR.) GRUPPE;B`, `LEUKOCYTTYPE GRUPPE;B`, `LEUKOCYTTYPE; ANTALK. (LISTE);B` (dropped by default config) |
| `TEG-R` / `TEG-MA` / `TEG-LY30` | same names |

Default aggregation: `max` (channels `LACTATE_max`, `HEMOGLOBIN_max`, …).

### 4.7 `Medicin` (medication administrations)

Filtered by `filter_medicin()`; becomes the multi-hot `medication` categorical channel (point events
at administration time).

| Column | Type/format | Meaning | Required |
|---|---|---|---|
| `CPR_hash` | str | Patient ID | yes |
| `Administrationstidspunkt` | datetime | Administration time (the event timestamp) | yes |
| `Seponeringstidspunkt` | datetime or empty | Discontinuation time (kept as `end`; not used for binning) | column recommended; values optional |
| `ATC` | str | Full ATC code, e.g. `N02AB02` | yes |
| `Handling` | str | Administration action — row kept **only if** in `MEDICATION_ACTION_LIST`: `Administreret`, `Ny pose`, `Selvadministration`, `Adm. ernæring/sterilt vand`, `Genstartet`, `Infusion/pose skiftet`, `Selvmedicinering`, `Status, indgift` | yes |
| `Administrationsdosis` | number | Dose — carried through for optional composite/profile features | no |
| `Dosisenhed` | str | Dose unit — same | no |

ATC prefixes → categories (`ATC_LVL3_MAP` / `ATC_LVL4_MAP`): level 3 — `cardiovascular_drugs`
(C01/C02/C07), `antibiotics` (J01), `neuro_drugs` (N05/M03), `anti_thrombotic` (B01), `diuretics`
(C03), `hemostatics` (B02), `hormone_drugs` (H01), `antidotes` (V03); level 4 — `infusion`
(B05B/B05X), `blood` (B05A), `opiods` (N02A), `local_anastethics` (N01B), `anastethics` (N01A),
`insulin` (A10A). Codes matching no prefix are ignored.

### 4.8 `Procedurer` (procedures) — only if enabled in the model's concept list

Filtered by `filter_procedures()`.

| Column | Type/format | Meaning | Required |
|---|---|---|---|
| `CPR_hash` | str | Patient ID | yes |
| `ServiceDatetime` | datetime | Procedure time | yes |
| `ProcedureCode` | str | Danish SKS procedure code; kept only if it starts with a prefix in `PROCEDURE_MAP` | yes |

Prefixes → categories: `KA` neuro, `KB` endokrin, `KC` øje, `KD` ønh, `KE` oral, `KF` kardio, `KG`
thorax, `KH` mamma, `KJ` abdomen, `KK` uro, `KL` gyn, `KM` obstetrik, `KN` orto, `KP` vaskulær,
`KQ` hud, `BGD` respirator, `BGA` sonde_tube.

### 4.9 `ITAOversigtsrapport` (ICU flowsheet scores)

Filtered by `filter_ita()`.

| Column | Type/format | Meaning | Required |
|---|---|---|---|
| `CPR_hash` | str | Patient ID | yes |
| `Målingstidspunkt` | datetime | Measurement time | yes |
| `ITAOversigt_Måling` | str | Measurement name; consumed: `GLASGOW COMA SCORE` / `Glasgow Coma Score` → `GCS`. (`SAPS 3 SCORE` → `SAPS3` and `SOFA total score` → `SOFA` are mapped but dropped by the default `drop_features` config.) | yes |
| `Værdi` | number | Score value | yes |

Default aggregation: `min` (channel `GCS_min`). GCS is additionally augmented from clinical notes
(section 4.11).

### 4.10 `EWS` (early warning score)

Filtered by `filter_ews()` **and** merged into vitals by `extract_ews_vitals()`
(both `astra/data/filters.py`). The merge happens inside `_filter_concepts_for_patient`
(`astra/inference/data_prep.py`), mirroring the training pipeline.

| Column | Type/format | Meaning | Required |
|---|---|---|---|
| `CPR_hash` | str | Patient ID | yes |
| `Målingstidspunkt` | datetime | Measurement time | yes |
| `EWS_Måling` | str | Measurement name (below) | yes |
| `Værdi` | str/number | Value; vital rows may carry the EWS sub-score in parentheses — `"120/70 (0)"` is parsed to `"120/70"` | yes |

`EWS_Måling` values consumed:

- `'EWS korr. total score'` → continuous channel `EWS_SCORE` (aggregation `max`).
- `'SAT (score)'` → `Saturation`, `'Temp. (score)'` → `Temperatur`, `'BT (score)'` → `BT`,
  `'RF (score)'` → `Resp.frekvens` — these rows are **converted into VitaleVaerdier rows** and merged
  into the vitals stream (deduplicated against native vitals on time+parameter+value).

### 4.11 `Notater` (clinical notes) — feeds three derived concepts

Loaded once per patient in `_filter_concepts_for_patient` and mined by
`astra/data/notes_features.py` + `astra/data/cardiac_arrest.py`.

| Column | Type/format | Meaning | Required |
|---|---|---|---|
| `CPR_hash` | str | Patient ID | yes |
| `Redigeringstidspunkt` | datetime | Note edit time (the timestamp used) | yes |
| `Note` | str | Free text | yes |
| `Notetype` | str | Note type — used to prioritize/exclude notes (e.g. intubation search order, `hjertestop` matching) | yes |

Derived features:

- **`ISS_notes`** — `build_iss_from_notes()`: regex-extracts Injury Severity Score mentions; keeps one
  value per patient (max value at earliest timestamp) → continuous channel `ISS_notes_max`.
- **GCS augmentation** — `build_gcs_from_notes()` adds GCS values found in text (`gcs|glasgow`) to
  `ITAOversigtsrapport`.
- **`Events`** (only if in the model's concept list) — `build_cardiac_arrest_from_notes()`
  (keyword `hjertestop`) + `build_intubation_from_notes()` (intubation within 24 h of admission) →
  categorical `event` channel.

### 4.12 `Diagnoser` (diagnosis history) — feeds the Elixhauser score

Consumed by `compute_elixhauser_for_patient()` (`astra/inference/comorbidity.py`) to produce the
static `ASMT_ELIX` feature (van Walraven-weighted Elixhauser, ICD-10 Quan mapping).

| Column | Type/format | Meaning | Required |
|---|---|---|---|
| `CPR_hash` | str | Patient ID | yes |
| `Diagnosekode` | str | Danish SKS diagnosis code: `'D'` + ICD-10 + one trailing modifier character (e.g. `DA105B`). The code is converted by stripping the **first and last character** (`.str.slice(1, -1)`) — supply codes in exactly this form. | yes |
| `Noteret_dato` | datetime | Date the diagnosis was noted | yes |
| `Løst_dato` | datetime or empty | Date resolved | column yes; values optional |

Only **pre-existing** diagnoses count: `Noteret_dato ≤ start − 1 day` AND (`Løst_dato` null OR
`≥ start + 1 day`). Include the patient's full diagnosis history, not just the current encounter.

### 4.13 `ISS_computed` — simplified data-source contract

In the training environment this comes from an R/ICDPIC-R pipeline. **External deployments provide it
directly** (or omit it). When a data source is registered it is authoritative
(`_filter_concepts_for_patient`, `astra/inference/data_prep.py` — the `ISS_computed` branch):

- `fetch(cpr_hash, 'ISS_computed')` returns a frame with a **numeric `VALUE` column** — the
  **first valid (non-NaN) value wins** — or `None` when unavailable.
- The value is timestamped at admission and fed to the continuous channel `ISS_computed_max`.

```python
def fetch(self, cpr_hash, concept):
    if concept == 'ISS_computed':
        iss = my_iss_lookup(cpr_hash)                 # your trauma-registry ISS
        return pd.DataFrame({'VALUE': [iss]}) if iss is not None else None
    ...
```

### 4.14 `fetch_prehospital` — optional capability

If (and only if) the model was trained with prehospital data (`prehospital: true` in the shipped
config), the bin grid starts at the **prehospital encounter**, not hospital arrival, and the static
ABCD assessment features exist. Your source may implement:

```python
def fetch_prehospital(self, cpr_hash: str) -> Optional[dict]:
    return {
        'prehospital_start': "2023-08-15 09:12:00",   # timestamp-like; may be None
        'A': 'Fri',              # airway        — optional
        'B': 'Normal',           # breathing     — optional
        'C': 'Let påvirket',     # circulation   — optional
        'D': 'Vågen',            # consciousness — optional
    }   # or None when the patient has no prehospital record
```

Handled by `_apply_prehospital_start()` (`astra/inference/data_prep.py`): `start` becomes
`min(prehospital_start, hospital_start)`, shifting the whole bin grid; A–D become static categorical
features. Category values must match the training vocabulary (`ABCD_SEVERITY` in
`astra/data/mappings.py`): A ∈ {Fri, Truede, Blokerede}; B ∈ {Normal, Let påvirket, Meget påvirket,
Respirationsstop}; C ∈ {…, Hjertestop}; D ∈ {Vågen, Bevidsthedspåvirket, Bevidstløs} — unknown values
are encoded as `#na#`.

### 4.15 `InvasiveMonitoring` — derived, never fetched

Built by `filter_vitals()` from the invasive vital parameter names in section 4.5 and passed via
`data/interim/concepts/InvasiveMonitoring.pkl`. Your source is never asked for it; just make sure the
invasive `Vital_parametre` names survive in your vitals feed and that `data/interim/concepts/` is
writable.

---

## 5. Degradation matrix

What happens when a concept/capability is absent (derived from the code paths cited in section 4 —
`None`/empty from `fetch` raises `FileNotFoundError` inside `load_patient_csv`, which each caller
handles as below):

| Absent | Behavioral consequence | Severity |
|---|---|---|
| `ADTHaendelser` | `PatientNotFoundError` (404) — no trajectory can be built | **Fatal per patient** |
| `PatientInfo` | `PatientNotFoundError` (404) — no AGE/SEX | **Fatal per patient** |
| `VitaleVaerdier` | All vitals channels empty (missing → 0 after normalization); static HEIGHT/WEIGHT `NaN` → mean-imputed with `_na` indicators; no `InvasiveMonitoring` events | High — flagship signal lost |
| `Labsvar` | Lab channels (`LACTATE_max`, `HEMOGLOBIN_max`, …) empty | High |
| `Medicin` | `medication` multi-hot categories all zero | Medium |
| `Procedurer` | `procedures` categories all zero (only if enabled in the model) | Medium |
| `ITAOversigtsrapport` | `GCS_min` only from notes (if `Notater` present), else empty | Medium |
| `EWS` | Warning logged; **vitals not augmented with EWS-sourced measurements**; `EWS_SCORE_max` empty | Medium |
| `Notater` | Warning `"notes-based features will be unavailable"`: no `ISS_notes`, no GCS-from-notes augmentation, no `Events` (cardiac arrest / intubation) | Medium |
| `ISS_computed` | Channel `ISS_computed_max` stays empty | Low–medium |
| `Diagnoser` | `ASMT_ELIX = 0.0` (logged) — comorbidity signal silently zeroed, **not** marked missing | Medium (silent) |
| `fetch_prehospital` not implemented / returns `None`, but model trained **with** prehospital | Bin grid anchored at **hospital arrival** instead of prehospital start (all bin positions shifted vs training); ABCD features default-encoded (`#na#`); warning `"bin grid may be misaligned"` when no batch base_df exists | ⚠️ **Accuracy risk — requires model-owner sign-off before go-live** |
| `ebm/` models dir, model expects `_ebm_pred` | Warning; `_ebm_pred` channel left empty; `explain_ebm` returns `None` | Medium — the model saw this channel in training |
| `calibrators/<M>/` | Predictions served uncalibrated (`calibration_method: null` in responses) | Low–medium |

> **Phase 1 note:** the first deployment phase does **not** use prehospital data — the shipped model
> is trained with `prehospital: false`, so the `fetch_prehospital` row above does not apply. It
> becomes relevant only if a later phase ships a prehospital-trained model.

General rule: a missing measurement is **not** an error — the model was trained on ~80 % missingness
and missing values are handled by design. A missing *systematic* source (whole concept) changes the
input distribution and should be quantified against the owner's golden patients before go-live
(section 10).

---

## 6. Python quickstart

> **Runnable version:** `python scripts/demo_api_usage.py` executes everything in
> sections 6–7 end-to-end with **zero artifacts and zero patient data** — it builds a
> tiny synthetic model and serves a synthetic patient through an `InMemoryDataSource`,
> exercising the exact production code path (including the REST layer if
> `fastapi` is installed). Start there.

```python
from astra.inference.api import AstraPredictor

predictor = AstraPredictor.load(
    config_path="handoff/configs/defaults.yaml",  # config-first: model_name + data-prep
    artifacts_dir="handoff",         # exported bundle root (artifacts under models/) —
                                     # a training-layout root like "models" works too
    device="cpu",
    data_source=MySQLDataSource(engine),   # your adapter — or omit for file-based mode
)
# ...or pass the model name explicitly: AstraPredictor.load("<MODEL>", artifacts_dir="handoff")

print(predictor.model_info()["channels"])

# Probability + curve at a point in time
resp = predictor.predict(
    patient_id="a3f9c2e8...",              # CPR_hash
    timestamp="2023-08-16 06:00",          # evaluation time
    service_date="2023-08-15",             # trauma admission date → selects the encounter
)
payload = resp.to_dict()                   # JSON-safe dict
print(payload["probability"], payload["eval_hours"])
print(payload["curve"]["hours"][:5], payload["curve"]["probabilities"][:5])

# SHAP explanation (everything the panels need)
shap = predictor.explain("a3f9c2e8...", "2023-08-16 06:00", service_date="2023-08-15", top_n=20)
shap_payload = shap.to_dict()

# What changed between hour 6 and hour 24?
diff = predictor.explain_differential("a3f9c2e8...", "2023-08-15", t1_hours=6.0, t2_hours=24.0)
```

**Offline smoke test without any real data** — the `InMemoryDataSource` plus the synthetic patient:

```python
import json
import pandas as pd
from astra.inference.api import AstraPredictor
from astra.inference.datasource import InMemoryDataSource
from astra.inference.synthetic import make_synthetic_raw_data   # same generator behind examples/synthetic_patient.json
                                                                # (see astra/inference/synthetic.py for the exact signature)

cpr, frames, prehosp = make_synthetic_raw_data()                # frames: {concept: DataFrame} in the section-4 schema
source = InMemoryDataSource({cpr: frames}, prehospital={cpr: prehosp} if prehosp else None)

predictor = AstraPredictor.load("<MODEL>", artifacts_dir="handoff", device="cpu", data_source=source)
resp = predictor.predict(cpr, "2023-08-16 06:00", service_date="2023-08-15")
assert 0.0 <= resp.probability <= 1.0
```

This is also exactly what `export_artifacts validate` runs — use it as the template for your own
integration tests: swap `InMemoryDataSource` for your SQL adapter, keep the assertions.

---

## 7. REST quickstart

The bundled reference service wraps `AstraPredictor` 1:1.

```bash
pip install -e .[service]

export ASTRA_CONFIG=configs/<cfg>.yaml   # config-first: supplies model_name + data-prep settings
export ASTRA_MODEL_NAME=<MODEL>          # optional override (default: model_name from ASTRA_CONFIG)
export ASTRA_ARTIFACTS_DIR=handoff       # bundle root (default: models)
export ASTRA_DATA_DIR=data/raw           # file-based mode only
export ASTRA_PATIENT_DIR=data/patients   # file-based mode only
export ASTRA_DEVICE=cpu                  # default: auto-detect
export ASTRA_CACHE_SIZE=8                # LRU patient-context cache
export ASTRA_LOG_LEVEL=INFO

python -m astra.service --port 8000
```

Settings are read by `ServiceSettings.from_env()` (`astra/service/settings.py`); request/response
bodies are the Pydantic models in `astra/service/schemas.py`, mirroring
`astra/inference/responses.py` field-for-field. (To serve from SQL rather than files, start the
service from a small launcher that calls `set_data_source(...)` /
`AstraPredictor.load(data_source=...)` before binding the port.)

```bash
curl -s localhost:8000/health
# {"status": "ok", "model_name": "<MODEL>", "model_loaded": true,
#  "is_temporal": true, "device": "cpu", "seq_len": 116}
# status "degraded" = service up but model failed to load (inference endpoints → 503)

curl -s localhost:8000/model/info | jq '.seq_len, .channels[:3], .calibration_method'

curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "patient_id": "a3f9c2e8...",
  "service_date": "2023-08-15",
  "timestamp": "2023-08-16 06:00"
}'
# optional body fields: "include_curve": true (predict), "top_n"/"include_values" (explain),
# "include_endpoints" (explain/differential)
```

```json
{
  "patient_id": "a3f9c2e8...",
  "pid": "a3f9c2e820230815",
  "model_name": "<MODEL>",
  "is_temporal": true,
  "survival_mode": false,
  "calibration_method": "isotonic",
  "admission_time": "2023-08-15T10:42:11",
  "requested_time": "2023-08-16T06:00:00",
  "eval_hours": 20.0,
  "eval_step": 61,
  "trajectory_length": 62,
  "seq_len": 116,
  "probability": 0.083,
  "curve": {
    "steps": [0, 1, 2, "..."],
    "hours": [0.167, 0.333, 0.5, "..."],
    "probabilities": [0.031, 0.030, 0.034, "..."],
    "source": "temporal_head",
    "survival": null
  },
  "inhospital_start_hours": null,
  "compute_ms": 412.7
}
```

```bash
curl -s -X POST localhost:8000/explain -H 'Content-Type: application/json' -d '{
  "patient_id": "a3f9c2e8...", "service_date": "2023-08-15",
  "timestamp": "2023-08-16 06:00", "top_n": 20
}'
# → ExplanationResponse (section 8): ts_shap [n_channels][seq_len], ts_values, cat_ts, static_cat,
#   static_cont, time_axis, channels, top_features, completeness, ...

curl -s -X POST localhost:8000/explain/differential -H 'Content-Type: application/json' -d '{
  "patient_id": "a3f9c2e8...", "service_date": "2023-08-15",
  "t1_hours": 6.0, "t2_hours": 24.0
}'

curl -s -X POST localhost:8000/explain/ebm -H 'Content-Type: application/json' -d '{
  "patient_id": "a3f9c2e8...", "service_date": "2023-08-15",
  "timestamp": "2023-08-16 06:00"
}'
# → null unless the model has the _ebm_pred channel
```

Error mapping: 404 unknown patient, 422 timestamp before admission, 400 unparseable timestamp/missing
`service_date`, 503 artifacts unavailable (section 1).

---

## 8. Response reference → SHAP panel mapping

All response objects live in `astra/inference/responses.py`; `to_dict()` yields plain JSON types.
The payloads deliberately carry everything needed to recreate every panel in `dashboard/app_shap.py`
(Streamlit tabs: *SHAP Heatmaps*, *SHAP Overview*, *Differential SHAP*, *Data Completeness*) without
knowing anything about the model or bin configuration.

| Response field(s) | Shape / type | Dashboard panel it powers |
|---|---|---|
| `PredictionResponse.curve` (`steps`, `hours`, `probabilities`, `source`, `survival`) + `eval_step` | lists, len = `trajectory_length` | Probability-over-time trajectory plot (`plot_prediction_trajectory_plotly`) with a marker at the evaluated step |
| `ExplanationResponse.ts_shap` + `channels` + `time_axis` (+ `channel_map` for grouping channels by source concept) | `[n_channels][seq_len]` | Continuous-TS SHAP heatmap (`plot_continuous_ts_shap_plotly`, part of `plot_unified_shap_heatmap_plotly`) |
| `ExplanationResponse.cat_ts` (`labels`, `shap_per_category`, `shap_aggregate`, `values_per_category`) | `[n_categories][seq_len]` | Categorical-TS SHAP heatmap (`plot_categorical_ts_shap_plotly`, part of unified heatmap) |
| `ExplanationResponse.static_cat` / `static_cont` (`names`, `shap`, `values`) | parallel lists | Static-feature SHAP bar chart (`plot_static_features_plotly`) |
| `ExplanationResponse.ts_values` + `trajectory_length` + `completeness` (`per_channel`, `overall`) | `[n_channels][seq_len]`, `null` = not measured | Data-completeness panel (`plot_data_completeness_plotly`) |
| `ExplanationResponse.top_features` (`[{name, importance}]`) | list of dicts | Top-channels bar chart (`plot_top_channels_plotly`); also SHAP budget/temporal overview (`plot_shap_budget_plotly`, `plot_shap_temporal_plotly`) |
| `ExplanationResponse.encoding_info` (`feature_ranges`, `category_labels`) | dict | Row labels/grouping for the categorical heatmap |
| `ExplanationResponse.inhospital_start_step` | int or null | Vertical "hospital arrival" marker (prehospital models) |
| `DifferentialExplanationResponse.delta_ts_shap` / `delta_cat_ts` / `delta_static_cat` / `delta_static_cont` + `top_delta_features`, `t1_probability`, `t2_probability` | same shapes as above | Differential SHAP heatmap + Δ-temporal plot (`plot_delta_shap_temporal_plotly`) and Δ top-channels/static bars |
| `explain_ebm(...)` dict | per-timeframe contributions | EBM contributions panel (`plot_ebm_contributions_plotly`) |

**`TimeAxis`** (`steps`, `hours_start`, `hours_end`, `bin_freq`) maps step index → elapsed hours since
admission (t=0). Frontends should always plot against `time_axis`, never assume uniform bins.

**Conventions:**

- **NaN → `null`.** `to_dict()` converts `NaN`/`±inf` to JSON `null` everywhere. In `ts_values`,
  `null` means *not measured in that bin* (that is the completeness signal). In `curve.probabilities`
  (simulation source), `null` means *no prediction computed at that bin*— carry the last non-null
  value forward when drawing.
- **Truncation to `trajectory_length`.** `curve` arrays are truncated to the visible trajectory.
  `ts_shap`/`ts_values`/`cat_ts` arrays are full `seq_len` wide, but positions `≥ trajectory_length`
  (and beyond the censored `eval_step`) are zeroed/padding — crop your heatmap x-axis to
  `trajectory_length` (the dashboard does).
- `eval_step`/`eval_hours` tell you which bin the probability and SHAP attribution refer to;
  `requested_time` echoes the caller's timestamp (which may have been clamped — section 9).
- Timestamps are ISO 8601 strings; `pid` is the deterministic inference PID
  (`make_inference_pid`: first 8 chars of `CPR_hash` + `YYYYMMDD` of `service_date`).

---

## 9. Operational notes

- **First call per patient is the expensive one.** It builds the `PatientContext`: fetches *all*
  concepts through your data source, builds the trajectory, mines notes, computes Elixhauser, bins and
  encodes. Wall time is dominated by your data fetch (each concept load is timed and logged:
  `"Concept load timing: ..."`). Subsequent calls hit the context cache.
- **LRU context cache.** Contexts are cached per `(patient_id, service_date)` key
  (`context_cache_size`, default 8; `ASTRA_CACHE_SIZE` in the service). Contexts only move **forward**
  in time: a later `timestamp` advances the cached context incrementally (only new bins are
  re-aggregated); an earlier `timestamp` is served from the same context by **censoring** — inputs
  after the requested step are masked, so results are identical to a fresh build at that time.
  Eviction is silent; the next request rebuilds.
- **Clamping semantics.** A `timestamp` beyond the patient's data extent is clamped to the trajectory
  end (`patient_end_time`) — you get the latest available prediction, with `eval_hours` reflecting the
  clamped step. A `timestamp` before admission raises `TimestampBeforeAdmissionError` (HTTP 422).
  `service_date` is mandatory — it selects the encounter when a patient has several.
- **SHAP cost.** `explain` runs `shap.GradientExplainer` against the bundle's background tensors
  (≤ 200 training samples) with `nsamples` = `evaluation.shap_nsamples` (default 200, seeded by
  `evaluation.shap_seed`) — expect seconds, not milliseconds, on CPU. `explain_differential` is ~2×
  an `explain` plus two predictions. Budget UI affordances accordingly (`compute_ms` is reported in
  every response).
- **Non-temporal replay cost.** For non-temporal models the first `predict` per patient replays the
  whole trajectory (one forward pass per bin). Advancing queries only pay for new bins; queries at or
  before the current position read the stored curve.
- **Single-worker constraint.** `AstraPredictor` holds one `threading.RLock` around all inference —
  the model and SHAP explainer are not re-entrant — and the data source is **process-global**
  (`set_data_source` in `astra/inference/patient_store.py`). Run one predictor per process; scale by
  running multiple processes (each loads its own model copy), not threads. Requests within a process
  are serialized.
- **Memory.** Model + bundle (incl. SHAP background) is a few hundred MB; each cached context holds
  its tensors plus accumulated raw events.
- **Logging.** Standard `logging` under the `astra.*` namespace; the service honours
  `ASTRA_LOG_LEVEL`. Patient identifiers are logged truncated to 8 characters.

---

## 10. Verification checklists

### (a) Owner — pre-handoff, on the secure (Azure) environment

Most of this is automated by the driver script (run from the repo root):

```bash
python -m pytest tests/ -q                                        # synthetic suite (87 tests)
python scripts/azure_handoff_check.py --config configs/<experiment>.yaml --sign-off "<approval>"
```

The driver runs: the synthetic export self-test, a real-model export + validate round trip,
golden-patient parity (facade vs `SimulationRunner`/`InferenceSession` on auto-picked holdout
patients: probabilities and curves to 1e-6, SHAP exact-or-correlation ≥ 0.95), and a
`run_inference` CLI smoke. Manual boxes:

- [ ] **`azure_handoff_check.py` RESULT: PASS** (attach the summary to the handoff).
- [ ] **Export + validate in a fresh checkout** (empty `data/interim/`, no cohort data) —
  proves no hidden dependency on interim files.
- [ ] **Prehospital degradation quantified** — only if a prehospital-trained model is ever shipped
  (phase 1 ships `prehospital: false`, so this is N/A initially).
- [ ] **manifest sign-off.** `sign_off` recorded (via `--sign-off` or by editing `manifest.json`)
  before the bundle leaves the secure environment (section 2).

### (b) Team — acceptance, in your environment

- [ ] **Bundle validation:** `python -m astra.inference.export_artifacts validate --dir handoff/`
  → PASS (hashes, model load, synthetic forward pass).
- [ ] **Service up:** `GET /health` returns `status: ok` with the expected `model_name`;
  `GET /model/info` shows the expected `seq_len`, channel list and `calibration_method`.
- [ ] **Synthetic predict:** `POST /predict` for the patient in `examples/synthetic_patient.json`
  (served through `InMemoryDataSource` or your adapter) returns `0 ≤ probability ≤ 1` and a curve of
  length `trajectory_length`.
- [ ] **Real patient predict:** one real patient through **your** `PatientDataSource`; verify
  `admission_time` matches your ADT source, `trajectory_length > 0`, per-concept log lines show the
  expected row counts, and the degradation matrix (section 5) explains every "concept absent" log.
- [ ] **SHAP smoke:** `POST /explain` on the same patient returns non-empty `ts_shap`,
  `completeness.overall > 0`, and `top_features` that are clinically plausible; render one heatmap
  from the payload alone (no model access) to prove the section-8 mapping.
- [ ] **Error paths:** unknown patient → 404; timestamp before admission → 422; timestamp far beyond
  discharge → 200 with clamped `eval_hours`.

---

*Where this document and the code disagree, the code wins — key sources:
`astra/inference/api.py`, `astra/inference/responses.py`, `astra/inference/datasource.py`,
`astra/inference/patient_store.py`, `astra/inference/data_prep.py`, `astra/inference/pipeline.py`,
`astra/inference/comorbidity.py`, `astra/data/filters.py`, `astra/data/mappings.py`.*
