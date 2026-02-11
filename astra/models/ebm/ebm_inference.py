import os
import pickle
import numpy as np

from astra.utils import logger
from generate_ebm_feature import (
    _create_aggregated_dataset,
    preprocess_features,
)


def load_deployment_models(
    save_dir: str = "data/interim/ebm_features",
):
    path = os.path.join(save_dir, "ebm_deployment_models.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def infer_new_patients(
    new_base_df,
    cfg_dict: dict,
    save_dir: str = "data/interim/ebm_features",
):
    """
    Generate EBM interval predictions for new patients.
    """

    deployment_models = load_deployment_models(save_dir)

    results = {}

    for masking_hours, model_dict in deployment_models.items():

        X_full, _, _, _ = _create_aggregated_dataset(
            new_base_df,
            cfg_dict,
            masking_hours,
        )

        id_col = cfg_dict["dataset"]["id_col"]
        pids = X_full[id_col].values
        X_features = X_full.drop(columns=[id_col])

        X_processed, _, _ = preprocess_features(
            X_features,
            model_dict["expected_cat_feats"],
            model_dict["expected_cont_feats"],
            encoder=model_dict["encoder"],
            fit=False,
            expected_cat_feats=model_dict["expected_cat_feats"],
            expected_cont_feats=model_dict["expected_cont_feats"],
        )

        probs = model_dict["model"].predict_proba(X_processed)[:, 1]

        for pid, prob in zip(pids, probs):
            results.setdefault(pid, {})[masking_hours] = float(prob)

    return results
