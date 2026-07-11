"""ASTRA inference API — runnable end-to-end demo for the implementing team.

Runs ANYWHERE with `pip install -e .` — no model artifacts, no patient data:
it builds a tiny randomly-initialized model + deployment bundle, serves a
synthetic patient through the `PatientDataSource` seam (exactly how your
SQL/FHIR/file adapter will), and walks through every API call your frontend
needs:

    python scripts/demo_api_usage.py            # full demo, synthetic model
    python scripts/demo_api_usage.py --json     # also dump full JSON payloads

Against real artifacts (secure environment), the identical calls apply —
swap the load line for::

    predictor = AstraPredictor.load(config_path="configs/defaults.yaml",
                                    artifacts_dir="models")

Sections:
  1. Build tiny artifacts (stand-in for the real handoff bundle)
  2. Implement a PatientDataSource (the ONE integration point you own)
  3. Load the predictor
  4. predict()            — probability + probability-over-time curve
  5. explain()            — SHAP payload; rebuilding the dashboard panels
  6. explain_differential() — what changed between two timepoints
  7. REST — the same via the bundled FastAPI app (in-process TestClient)

See docs/HANDOFF.md for the normative per-concept data contract.
"""

import argparse
import json
import logging
import os
import sys
import tempfile

import numpy as np
import pandas as pd

# Runnable from a bare checkout without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger('astra.scripts.demo')

CPR = "demo1234" + "0" * 56           # 64-char stand-in for a real CPR hash
SERVICE_DATE = "2030-01-01"
ADMISSION = pd.Timestamp("2030-01-01 12:00:00")


def banner(title):
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# 2. The data source — the one component the receiving team implements.
#    fetch() returns per-concept DataFrames in the raw EHR schema
#    (docs/HANDOFF.md §4). Here they are hardcoded; in production they come
#    from your SQL/FHIR/file backend.
# ---------------------------------------------------------------------------

def make_demo_patient_frames():
    """One synthetic trauma admission in the raw concept schema."""
    t = lambda hours: (ADMISSION + pd.Timedelta(hours=hours)).isoformat()

    patient_info = pd.DataFrame({
        "CPR_hash": [CPR],
        "Fødselsdato": ["1970-03-02"],
        "Dødsdato": [None],
        "Køn": ["Mand"],
    })

    adt = pd.DataFrame({
        "CPR_hash": [CPR, CPR, CPR],
        "ADT_haendelse": ["Indlæggelse", "Flyt Ind", "Udskrivning"],
        "Flyt_ind": [t(0), t(3), t(48)],
        "Flyt_ud": [t(3), t(48), t(48)],
        "Afsnit": ["RH TRAUMECENTER", "RH INTENSIV 4131", "RH INTENSIV 4131"],
    })

    def vital(hours, name, value):
        return {"CPR_hash": CPR, "Registreringstidspunkt": t(hours),
                "Vital_parametre": name, "Værdi": value}

    vitals = pd.DataFrame(
        [vital(h, "Puls", str(88 + i * 6)) for i, h in enumerate([0.2, 0.7, 1.5, 3.0, 6.0])]
        + [vital(h, "BT", f"{118 - i * 8}/{76 - i * 4}") for i, h in enumerate([0.3, 1.0, 2.5, 5.0])]
        + [vital(h, "Saturation", str(97 - i)) for i, h in enumerate([0.2, 1.2, 4.0])]
        + [vital(h, "Temperatur", "36.8") for h in [0.5, 6.0]]
        + [vital(0.4, "Højde", "70"),      # inches (converted to cm internally)
           vital(0.4, "Vægt", "2800")]     # ounces (converted to kg internally)
    )

    labs = pd.DataFrame([
        {"CPR_hash": CPR, "Prøvetagningstidspunkt": t(0.5),
         "BestOrd": "LAKTAT;P(AB)", "Resultatværdi": "3,1"},   # decimal comma OK
        {"CPR_hash": CPR, "Prøvetagningstidspunkt": t(4.0),
         "BestOrd": "LAKTAT;P(AB)", "Resultatværdi": "1,9"},
        {"CPR_hash": CPR, "Prøvetagningstidspunkt": t(1.0),
         "BestOrd": "HÆMOGLOBIN;B", "Resultatværdi": "7.8"},
    ])

    return {
        "PatientInfo": patient_info,
        "ADTHaendelser": adt,
        "VitaleVaerdier": vitals,
        "Labsvar": labs,
        # Concepts you don't serve are simply absent — the pipeline degrades
        # per the matrix in HANDOFF.md §5 (e.g. no Diagnoser -> ASMT_ELIX=0).
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true",
                        help="Print full JSON payloads (verbose)")
    parser.add_argument("--artifacts-dir", default=None,
                        help="Where to write the tiny demo artifacts "
                             "(default: a temp dir)")
    args = parser.parse_args()

    from astra.utils import setup_logging
    setup_logging(level=logging.WARNING)   # keep the demo output readable
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="astra.data.filters")

    # -----------------------------------------------------------------
    banner("1. Artifacts — tiny random model standing in for the real bundle")
    # -----------------------------------------------------------------
    from astra.inference.synthetic import save_tiny_artifacts

    artifacts_dir = args.artifacts_dir or tempfile.mkdtemp(prefix="astra_demo_")
    save_tiny_artifacts(artifacts_dir, model_name="demomodel")
    print(f"Tiny bundle + weights written to: {artifacts_dir}")
    print("(with the real handoff bundle you skip this step — you already "
          "have models/deployment/deployment_<M>.pkl and <M>.pth)")

    # -----------------------------------------------------------------
    banner("2. Data source — the ONE integration point you implement")
    # -----------------------------------------------------------------
    from astra.inference.datasource import InMemoryDataSource

    frames = make_demo_patient_frames()
    source = InMemoryDataSource({CPR: frames})
    print("PatientDataSource.fetch(cpr_hash, concept) -> DataFrame in the raw")
    print("EHR schema. This demo serves from memory; yours queries SQL/FHIR/files.")
    for concept, df in frames.items():
        print(f"  {concept:<16} {len(df):>2} rows   columns: {list(df.columns)}")

    # -----------------------------------------------------------------
    banner("3. Load the predictor")
    # -----------------------------------------------------------------
    from astra.inference.api import AstraPredictor

    predictor = AstraPredictor.load(
        "demomodel",
        artifacts_dir=artifacts_dir,
        device="cpu",
        data_source=source,
    )
    info = predictor.model_info()
    print(f"model={info['model_name']}  temporal={info['is_temporal']}  "
          f"seq_len={info['seq_len']}")
    print(f"channels: {info['channels']}")
    print(f"time axis (first 3 bins, hours): "
          f"{list(zip(info['time_axis']['hours_start'], info['time_axis']['hours_end']))[:3]}")

    # -----------------------------------------------------------------
    banner("4. predict() — probability + probability-over-time")
    # -----------------------------------------------------------------
    eval_time = ADMISSION + pd.Timedelta(hours=6)
    resp = predictor.predict(CPR, eval_time, SERVICE_DATE)
    p = resp.to_dict()
    print(f"P(deceased_30d) = {p['probability']:.4f} at t={p['eval_hours']:.1f}h "
          f"(step {p['eval_step']}, trajectory_length={p['trajectory_length']})")
    curve = p["curve"]
    print(f"curve source: {curve['source']}   points: {len(curve['probabilities'])}")
    print("hours:         ", [f"{h:.2f}" for h in curve["hours"][:6]], "...")
    print("probabilities: ", [None if v is None else f"{v:.3f}"
                              for v in curve["probabilities"][:6]], "...")
    if args.json:
        print(json.dumps(p, indent=2)[:2000])

    # -----------------------------------------------------------------
    banner("5. explain() — the SHAP payload behind every dashboard panel")
    # -----------------------------------------------------------------
    expl = predictor.explain(CPR, eval_time, SERVICE_DATE, top_n=5)
    e = expl.to_dict()
    print(f"explains the prediction at step {e['eval_step']} "
          f"(t={e['eval_hours']:.1f}h)")
    print("top features:", [(f['name'], round(f['importance'], 4))
                            for f in e['top_features']])
    print(f"completeness: overall={e['completeness']['overall']:.2f}  "
          f"per channel={ {k: round(v, 2) for k, v in e['completeness']['per_channel'].items()} }")

    # Rebuilding the continuous-TS SHAP heatmap (dashboard parity):
    shap_matrix = np.array([[0.0 if v is None else v for v in row]
                            for row in e["ts_shap"]])
    print("\nSHAP heatmap ingredients (HANDOFF.md §8):")
    print(f"  y axis: e['channels']            -> {e['channels']}")
    print(f"  x axis: e['time_axis']['hours_end'][:trajectory_length]")
    print(f"  z:      e['ts_shap']             -> matrix {shap_matrix.shape}")
    print(f"  tooltips/raw values: e['ts_values'] (None = not measured)")
    row = shap_matrix[0][: e["trajectory_length"]]
    print(f"  e.g. channel '{e['channels'][0]}' attribution over time: "
          f"{[round(v, 4) for v in row[:6]]} ...")
    if e["static_cont"]:
        print(f"  static features: {list(zip(e['static_cont']['names'], e['static_cont']['values']))}")
    if args.json:
        print(json.dumps(e, indent=2)[:2000])

    # -----------------------------------------------------------------
    banner("6. explain_differential() — what changed between T1 and T2")
    # -----------------------------------------------------------------
    diff = predictor.explain_differential(CPR, SERVICE_DATE,
                                          t1_hours=1.0, t2_hours=2.5)
    d = diff.to_dict()
    print(f"P: {d['t1_probability']:.4f} (t={d['t1_hours']:g}h, step {d['t1_step']}) "
          f"-> {d['t2_probability']:.4f} (t={d['t2_hours']:g}h, step {d['t2_step']})")
    print("top delta features:", [(f['name'], round(f['importance'], 4))
                                  for f in d['top_delta_features'][:5]])

    # -----------------------------------------------------------------
    banner("7. REST — identical payloads via the bundled FastAPI reference")
    # -----------------------------------------------------------------
    try:
        from fastapi.testclient import TestClient
        from astra.service.app import create_app
    except ImportError:
        print("fastapi not installed — `pip install -e .[service]` to run the")
        print("service: python -m astra.service --port 8000, then e.g.:")
        print(f"""  curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \\
    -d '{{"patient_id": "{CPR[:16]}...", "service_date": "{SERVICE_DATE}", "timestamp": "2030-01-01 18:00"}}'""")
        return

    client = TestClient(create_app(predictor=predictor))
    print("GET  /health       ->", client.get("/health").json())
    body = {"patient_id": CPR, "service_date": SERVICE_DATE,
            "timestamp": "2030-01-01 18:00:00"}
    r = client.post("/predict", json=body)
    print(f"POST /predict      -> {r.status_code}, "
          f"probability={r.json()['probability']:.4f}")
    r = client.post("/explain", json={**body, "top_n": 3})
    print(f"POST /explain      -> {r.status_code}, "
          f"top={[(f['name']) for f in r.json()['top_features']]}")
    # Error contract:
    r = client.post("/predict", json={**body, "timestamp": "2029-12-31 00:00"})
    print(f"POST /predict (timestamp before admission) -> {r.status_code} "
          f"({r.json()['detail'][:60]}...)")

    print("\nDemo complete. Swap section 1+2 for the real bundle and your own")
    print("PatientDataSource, and everything from section 3 on is production code.")


if __name__ == "__main__":
    main()
