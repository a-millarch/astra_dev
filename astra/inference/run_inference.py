"""Single-patient inference CLI — thin wrapper over :class:`AstraPredictor`.

Run in an environment where model artifacts (and a data source) exist:

    python -m astra.inference.run_inference \
        --model-name <MODEL> \
        --patient-id <CPR_HASH> \
        --service-date "2023-08-15" \
        --timestamp "2023-08-16 06:00:00" \
        [--explain] [--differential 6 24] [--ebm] \
        [--out reports/inference/result.json] [--plot] [--verbose]

Outputs a JSON document with the prediction (probability + probability-over-
time curve) and, when requested, the SHAP explanation payloads — the same
structures the REST service (``python -m astra.service``) returns.

For interactive figure-based inspection (notebooks, simulation), see
:func:`astra.visualize.inference.default_session_plot`.
"""

import argparse
import json
import logging
import os

from astra.inference.api import AstraPredictor

# Backward compatibility: default_session_plot lived here historically.
from astra.visualize.inference import default_session_plot  # noqa: F401

logger = logging.getLogger(__name__)


def run(model_name, patient_id, service_date, timestamp, *,
        config_path=None, artifacts_dir='models', data_dir='data/raw',
        patient_dir='data/patients', device=None,
        explain=False, differential=None, ebm=False,
        out_path=None, plot=False, save_dir='reports/inference'):
    """Execute one prediction (and optional explanations); return the payload dict.

    ``model_name`` may be None when ``config_path`` points to a YAML with a
    ``model_name`` key (config-first, like the training CLI).
    """
    predictor = AstraPredictor.load(
        model_name,
        artifacts_dir=artifacts_dir,
        config_path=config_path,
        device=device,
        data_dir=data_dir,
        patient_dir=patient_dir,
    )
    logger.info("Model '%s' loaded (temporal=%s, seq_len=%d)",
                predictor.model_name, predictor.is_temporal, predictor.seq_len)

    prediction = predictor.predict(patient_id, timestamp, service_date)
    logger.info("P(deceased_30d)=%.4f at t=%.1fh (step %d, traj_len=%d)",
                prediction.probability, prediction.eval_hours,
                prediction.eval_step, prediction.trajectory_length)

    payload = {'prediction': prediction.to_dict()}

    if explain:
        logger.info("Computing SHAP explanation...")
        explanation = predictor.explain(patient_id, timestamp, service_date)
        payload['explanation'] = explanation.to_dict()
        top = ", ".join(f"{f['name']}={f['importance']:.3f}"
                        for f in explanation.top_features[:5])
        logger.info("Top features: %s", top)

    if differential is not None:
        t1, t2 = differential
        logger.info("Computing differential SHAP (T1=%.1fh, T2=%.1fh)...", t1, t2)
        diff = predictor.explain_differential(patient_id, service_date, t1, t2)
        payload['differential'] = diff.to_dict()
        logger.info("delta P = %+.4f (%.4f -> %.4f)",
                    diff.t2_probability - diff.t1_probability,
                    diff.t1_probability, diff.t2_probability)

    if ebm:
        ebm_expl = predictor.explain_ebm(patient_id, timestamp, service_date)
        if ebm_expl is None:
            logger.info("EBM explanations not available for this model")
        else:
            payload['ebm'] = ebm_expl

    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info("Saved result to %s", out_path)

    if plot:
        _plot_curve(prediction, save_dir)

    return payload


def _plot_curve(prediction, save_dir):
    """Save a simple probability-over-time figure from the response payload."""
    if prediction.curve is None:
        logger.warning("No probability curve in the response — nothing to plot")
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    hours = prediction.curve.hours
    probs = [p if p is not None else float('nan')
             for p in prediction.curve.to_dict()['probabilities']]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hours, probs, 'o-', color='steelblue', markersize=3, linewidth=1.5)
    ax.axvline(prediction.eval_hours, color='crimson', linestyle=':',
               label=f'evaluated at {prediction.eval_hours:.1f}h')
    ax.set_xlabel('Hours since admission')
    ax.set_ylabel('P(deceased 30d)')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f'Prediction trajectory — {prediction.pid} '
                 f'({prediction.curve.source})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f'trajectory_{prediction.pid}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved trajectory plot to %s", out)


def main():
    parser = argparse.ArgumentParser(
        description="Single-patient inference: prediction + SHAP as JSON")
    parser.add_argument("--config", default=None,
                        help="Config YAML (config-first: model_name and data-prep "
                             "settings come from here; default configs/defaults.yaml)")
    parser.add_argument("--model-name", default=None,
                        help="Model name (default: model_name from --config)")
    parser.add_argument("--patient-id", "--cpr-hash", dest="patient_id",
                        required=True, help="Patient CPR hash")
    parser.add_argument("--service-date", required=True,
                        help="Trauma admission date (identifies the encounter)")
    parser.add_argument("--timestamp", "--current-time", dest="timestamp",
                        required=True, help="Evaluation time")
    parser.add_argument("--artifacts-dir", default="models",
                        help="Artifacts root (weights, deployment/, calibrators/, ebm/)")
    parser.add_argument("--data-dir", default="data/raw", help="Raw CSV directory")
    parser.add_argument("--patient-dir", default="data/patients",
                        help="Pre-split per-patient CSV directory")
    parser.add_argument("--device", default=None,
                        help="Force device (default: auto-detect)")
    parser.add_argument("--explain", action="store_true",
                        help="Include the SHAP explanation payload")
    parser.add_argument("--differential", nargs=2, type=float, default=None,
                        metavar=("T1", "T2"),
                        help="Include differential SHAP between two elapsed hours")
    parser.add_argument("--ebm", action="store_true",
                        help="Include EBM explanations (if the model uses them)")
    parser.add_argument("--out", default=None, help="Write the JSON payload here")
    parser.add_argument("--plot", action="store_true",
                        help="Save a trajectory PNG to --save-dir")
    parser.add_argument("--save-dir", default="reports/inference",
                        help="Output directory for figures")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable DEBUG-level logging")
    args = parser.parse_args()

    from astra.utils import setup_logging
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.model_name is None and args.config is None:
        # config-first default: derive model_name from configs/defaults.yaml
        args.config = "configs/defaults.yaml"

    run(
        model_name=args.model_name,
        patient_id=args.patient_id,
        service_date=args.service_date,
        timestamp=args.timestamp,
        config_path=args.config,
        artifacts_dir=args.artifacts_dir,
        data_dir=args.data_dir,
        patient_dir=args.patient_dir,
        device=args.device,
        explain=args.explain,
        differential=args.differential,
        ebm=args.ebm,
        out_path=args.out,
        plot=args.plot,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
