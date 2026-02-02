import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

from astra.utils import get_base_df, cfg, logger


from astra.data.datasets import AggregatedDS

def temporal_train_val_split(X, y, val_frac=0.25):
    n = len(X)
    split_idx = int((1 - val_frac) * n)

    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:]

    y_train = np.asarray(y[:split_idx])
    y_val = np.asarray(y[split_idx:])

    return X_train, X_val, y_train, y_val


def main(cfg):
    base_df=get_base_df()
    # Create aggregated dataset with 24-hour masking
    agg_ds = AggregatedDS(
        cfg=cfg,
        base_df=base_df,
        masking_point='2h',  
        agg_funcs=['first', 'last', 'min', 'max', 'mean', 'std'],
        concepts=cfg["concepts"],
        default_mode=True,
      #  use_gpu=False,
    )

    X, y = agg_ds.get_X_y()
    no_preproces = True

    model_X = X.copy(deep=True)
    #model_X = model_X.replace(0.0,-1000)


    categorical_features = agg_ds.categorical_features
    continuous_features = agg_ds.continuous_features

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent"))
        ]
    )

    continuous_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        #   ("scaler", StandardScaler())
        ]
    )
    if no_preproces:
        logger.info("passthrough preprocessing")
        categorical_pipeline = 'passthrough'
        continuous_pipeline = 'passthrough'
        
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_features),
            ("cont", continuous_pipeline, continuous_features),
        ],
        remainder="drop"
    )

    X_train, X_val, y_train, y_val = temporal_train_val_split(
        model_X, y, val_frac=0.20
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_val)

    # EBM needs feature names in correct order
    feature_names = categorical_features + continuous_features

    ebm = ExplainableBoostingClassifier(
        feature_names=feature_names,
        random_state=42,
        interactions ="3x",
        validation_size=0.2,
        early_stopping_rounds = 100, # can do 200
        max_leaves = 2, #3 
        inner_bags = 0, #Compute heavy!
        
    )

    ebm.fit(X_train_proc, y_train)


# Get predictions (assuming you have ebm_clf fitted)
y_proba = ebm.predict_proba(X_test_proc)[:, 1]  # Probability of positive class
y_pred = ebm.predict(X_test_proc)

# Binarize y_test (ensure 0/1 format)
y_test_bin = np.array(y_val).round().astype(int)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test_bin, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 5))

# ROC subplot
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test_bin, y_proba)
pr_auc = average_precision_score(y_test_bin, y_proba)

plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"ROC AUC: {roc_auc:.3f}")
print(f"PR AUC (AP): {pr_auc:.3f}")