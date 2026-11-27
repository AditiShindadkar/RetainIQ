"""Step 4: train and compare four models, pick the best, explain it with SHAP.

How the best model is chosen
----------------------------
Missing a churner (false negative) costs more than calling a customer who was
going to stay anyway (false positive). So we rank models by **F2 score**, a
version of F1 that weighs recall twice as much as precision.

The ranking uses 5-fold cross-validation on the TRAINING data only. The test
set is kept untouched for the final scorecard, so we never pick a model by
peeking at the test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, fbeta_score, make_scorer,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config

# Why only `tenure` as a number?
#   * TotalCharges ~ tenure x MonthlyCharges (corr with tenure = 0.83).
#   * MonthlyCharges is almost exactly the sum of the services the customer
#     has (R^2 = 0.999 when regressed on the service columns).
#   * num_addons is a count of the add-on columns already in the model.
# Keeping duplicates made logistic regression split their effect in confusing
# ways ("higher bill lowers churn"). Dropping them changed CV F2 by only
# -0.004 (well inside fold-to-fold noise) and made the explanations trustworthy.
NUMERIC_FEATURES = ["tenure"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

F2 = make_scorer(fbeta_score, beta=2)


def split(df: pd.DataFrame):
    X, y = df[FEATURES], df[config.TARGET]
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def candidate_models(pos_weight: float) -> dict:
    """All models get class balancing: only ~27% of customers churn, and
    without it the models would learn to mostly predict 'stays'."""
    rs = config.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rs),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=rs),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
            n_jobs=-1, random_state=rs),
        "LightGBM": LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            class_weight="balanced", verbose=-1, random_state=rs),
    }


def build_pipeline(model) -> Pipeline:
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


def score(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "f2": fbeta_score(y_true, pred, beta=2),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def compare_models(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = split(df)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    rows, fitted = [], {}
    for name, model in candidate_models(pos_weight).items():
        pipe = build_pipeline(model)
        cv_f2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=F2)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        rows.append({"model": name, **score(y_test, proba),
                     "cv_f2_mean": cv_f2.mean(), "cv_f2_std": cv_f2.std()})
        fitted[name] = pipe

    table = pd.DataFrame(rows).sort_values("cv_f2_mean", ascending=False).reset_index(drop=True)
    best_name = table.loc[0, "model"]
    return table, best_name, fitted, (X_train, X_test, y_train, y_test)


def out_of_fold_scores(df: pd.DataFrame, best_name: str) -> np.ndarray:
    """Churn score for EVERY customer, where each score comes from a model that
    never saw that customer during training (5-fold). This stops the risk groups
    from looking better than they really are."""
    y = df[config.TARGET]
    pos_weight = (y == 0).sum() / (y == 1).sum()
    pipe = build_pipeline(candidate_models(pos_weight)[best_name])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    return cross_val_predict(pipe, df[FEATURES], y, cv=cv, method="predict_proba")[:, 1]


def fit_final(df: pd.DataFrame, best_name: str) -> Pipeline:
    """Refit the winning model on all data for use in the dashboard."""
    y = df[config.TARGET]
    pos_weight = (y == 0).sum() / (y == 1).sum()
    pipe = build_pipeline(candidate_models(pos_weight)[best_name])
    return pipe.fit(df[FEATURES], y)


# ---------------------------------------------------------------- explanations
LABELS = {
    "tenure": "months as a customer (tenure)",
    "MonthlyCharges": "monthly bill",
    "TotalCharges": "total amount billed so far",
    "num_addons": "number of add-on services",
    "SeniorCitizen": "senior citizen",
    "Partner": "has a partner",
    "Dependents": "has dependents",
    "PhoneService": "has phone service",
    "MultipleLines": "multiple phone lines",
    "InternetService": "internet type",
    "OnlineSecurity": "online security add-on",
    "OnlineBackup": "online backup add-on",
    "DeviceProtection": "device protection add-on",
    "TechSupport": "tech support add-on",
    "StreamingTV": "streaming TV",
    "StreamingMovies": "streaming movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless billing",
    "PaymentMethod": "payment method",
    "gender": "gender",
}


def _readable(feature: str) -> tuple[str, bool]:
    """'cat__Contract_Month-to-month' -> ('contract = Month-to-month', True)."""
    kind, name = feature.split("__", 1)
    if kind == "num":
        return LABELS.get(name, name), False
    for col in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        if name.startswith(col + "_"):
            return f"{LABELS.get(col, col)} = {name[len(col) + 1:]}", True
    return name, True


def shap_importance(pipe: Pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame, top_n: int = 10):
    """Mean |SHAP| per feature, plus whether the feature pushes risk up or down."""
    import shap

    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = prep.get_feature_names_out()
    Xb = pd.DataFrame(prep.transform(X_background), columns=names)
    Xe = pd.DataFrame(prep.transform(X_explain), columns=names)

    if isinstance(model, LogisticRegression):
        values = shap.LinearExplainer(model, Xb).shap_values(Xe)
    else:
        values = shap.TreeExplainer(model).shap_values(Xe)
    values = np.asarray(values)
    if isinstance(values, list) or values.ndim == 3:  # some versions return one array per class
        values = values[..., 1] if values.ndim == 3 else values[1]

    rows = []
    for i, feat in enumerate(names):
        label, is_dummy = _readable(feat)
        col_vals, col_shap = Xe.iloc[:, i].to_numpy(), values[:, i]
        if is_dummy:
            effect = col_shap[col_vals > 0.5].mean() if (col_vals > 0.5).any() else 0.0
            raises = effect > 0
            sentence = f"{label[0].upper() + label[1:]} -> {'raises' if raises else 'lowers'} churn risk"
        else:
            corr = np.corrcoef(col_vals, col_shap)[0, 1] if col_vals.std() > 0 else 0.0
            raises = corr > 0
            sentence = f"Higher {label} -> {'raises' if raises else 'lowers'} churn risk"
        rows.append({"feature": feat, "plain_english": sentence,
                     "mean_abs_shap": float(np.abs(col_shap).mean()),
                     "direction": "raises risk" if raises else "lowers risk"})

    imp = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return imp.head(top_n), values, Xe
