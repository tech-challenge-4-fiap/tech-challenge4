#!/usr/bin/env python3
# scripts/evaluate_models.py
"""
Avalia modelos .joblib/.pkl em ./models usando o dataset Obesity.csv na raiz do repo
Gera results/metrics.md com um sumário.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from typing import Tuple, Optional

MODEL_DIR = "models"
DATA_PATHS = ["Obesity.csv", "obesity.csv", "./Obesity.csv"]
RESULTS_DIR = "results"
TARGET_CANDIDATES = ["Obesity_level", "Obesity level", "ObesityLevel", "Obesity_level"]


def find_data() -> Optional[str]:
    for p in DATA_PATHS:
        if os.path.exists(p):
            return p
    return None


def find_target_col(df: pd.DataFrame) -> Optional[str]:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    # last resort: look for any column containing 'obes' (case-insensitive)
    for c in df.columns:
        if "obes" in c.lower():
            return c
    return None


def prepare_X_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    # simple cleaning: drop non-informative columns and one-hot encode categoricals
    X = X.replace({np.nan: None})
    X = pd.get_dummies(X, drop_first=False)
    return X, y


def _resolve_loaded_object(loaded):
    """
    Resolve the object returned by joblib.load.
    - If it's a dict, try to extract 'model'/'estimator' and 'feature_cols'/'feature_names'.
    - Return (estimator, feature_names_from_file)
    """
    feature_names = None
    estimator = None

    if isinstance(loaded, dict):
        # common keys for feature names
        for k in ("feature_cols", "feature_names", "cols", "columns"):
            if k in loaded:
                try:
                    feature_names = list(loaded[k])
                except Exception:
                    feature_names = None
                break
        # common keys for model object
        for k in ("model", "estimator", "clf", "pipeline"):
            if k in loaded and hasattr(loaded[k], "predict"):
                estimator = loaded[k]
                break
        # fallback: look for any value that looks like an estimator
        if estimator is None:
            for v in loaded.values():
                if hasattr(v, "predict"):
                    estimator = v
                    break
    else:
        estimator = loaded

    return estimator, feature_names


def _extract_feature_names_from_estimator(estimator) -> Optional[list]:
    """
    Try several strategies to get feature names from the estimator/pipeline.
    """
    # direct attribute
    if hasattr(estimator, "feature_names_in_"):
        try:
            return list(getattr(estimator, "feature_names_in_"))
        except Exception:
            pass

    # pipeline: try to inspect named_steps and find a step with feature_names_in_
    if hasattr(estimator, "named_steps"):
        try:
            for step in estimator.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    return list(getattr(step, "feature_names_in_"))
        except Exception:
            pass

    # calibrated classifier wrapping a prefit estimator (CalibratedClassifierCV)
    # try to access underlying estimator if possible
    if hasattr(estimator, "estimator"):  # e.g., CalibratedClassifierCV has .estimator_
        try:
            inner = getattr(estimator, "estimator", None)
            if inner and hasattr(inner, "feature_names_in_"):
                return list(inner.feature_names_in_)
        except Exception:
            pass

    return None


def evaluate_model(model_path: str, X: pd.DataFrame, y: pd.Series, model_name: str = None) -> dict:
    loaded = joblib.load(model_path)
    if model_name is None:
        model_name = os.path.basename(model_path)

    estimator, feature_names_from_file = _resolve_loaded_object(loaded)

    if estimator is None or not hasattr(estimator, "predict"):
        return {"error": "Loaded object does not contain a usable estimator with .predict()", "model_name": model_name}

    # split
    stratify = y if (len(np.unique(y)) > 1 and len(y) > 50) else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

    # determine feature names to align
    feature_names = feature_names_from_file or _extract_feature_names_from_estimator(estimator)

    # align columns if feature names known
    X_test_aligned = X_test
    missing = []
    extra = []
    if feature_names is not None:
        if isinstance(X_test, pd.DataFrame):
            missing = [c for c in feature_names if c not in X_test.columns]
            extra = [c for c in X_test.columns if c not in feature_names]
            X_test_aligned = X_test.reindex(columns=feature_names, fill_value=0)
        else:
            # if X_test is numpy array, attempt conversion
            try:
                X_test_aligned = pd.DataFrame(X_test, columns=feature_names)
            except Exception:
                # fallback: keep original X_test
                X_test_aligned = X_test

    # final predict attempt
    try:
        y_pred = estimator.predict(X_test_aligned)
    except Exception as e:
        # try using the loaded object if estimator was nested inside a dict and a top-level pipeline exists
        return {"error": f"Prediction failed: {e}", "model_name": model_name}

    metrics = {}
    try:
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
        metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    except Exception as e:
        return {"error": f"Metrics computation failed: {e}", "model_name": model_name}

    metrics["model_name"] = model_name
    metrics["n_test"] = len(y_test)
    if feature_names is not None:
        metrics["feature_names_used"] = feature_names
        metrics["missing_features"] = missing
        metrics["extra_features"] = extra
    return metrics


def save_results(all_metrics: list, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Model evaluation results\n\n")
        f.write("Generated by `scripts/evaluate_models.py`\n\n")
        for m in all_metrics:
            f.write(f"## Model: `{m.get('model_name')}`\n\n")
            if "error" in m:
                f.write(f"**ERROR:** {m['error']}\n\n")
                continue
            f.write(f"- Test set size: {m.get('n_test')}\n")
            f.write(f"- Accuracy: **{m.get('accuracy'):.4f}**\n")
            f.write(f"- Balanced accuracy: **{m.get('balanced_accuracy'):.4f}**\n\n")
            # classification report
            f.write("### Classification report\n\n")
            cr = m.get("classification_report", {})
            f.write("| class | precision | recall | f1-score | support |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for label, vals in cr.items():
                if label in ("accuracy", "macro avg", "weighted avg"):
                    continue
                f.write(f"| {label} | {vals.get('precision',0):.3f} | {vals.get('recall',0):.3f} | {vals.get('f1-score',0):.3f} | {int(vals.get('support',0))} |\n")
            # add summary lines
            if "accuracy" in cr:
                f.write(f"\n- Accuracy (sklearn summary): {cr.get('accuracy')}\n")
            f.write("\n### Confusion matrix\n\n")
            cm = m.get("confusion_matrix", [])
            for row in cm:
                f.write("| " + " | ".join(str(x) for x in row) + " |\n")
            # log feature alignment info if present
            if "feature_names_used" in m:
                f.write("\n### Feature alignment\n\n")
                f.write(f"- feature_names_used: {m.get('feature_names_used')}\n")
                f.write(f"- missing_features: {m.get('missing_features')}\n")
                f.write(f"- extra_features: {m.get('extra_features')}\n")
            f.write("\n---\n\n")
        f.write("\n> End of report\n")
    print(f"Wrote results to {out_path}")


def main():
    data_path = find_data()
    if not data_path:
        print("Could not find dataset. Place Obesity.csv in the repo root.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    target_col = find_target_col(df)
    if target_col is None:
        print("Could not find target column (looked for Obesity_level). Columns found:", df.columns.tolist())
        sys.exit(1)

    X, y = prepare_X_y(df, target_col)

    model_files = []
    if os.path.isdir(MODEL_DIR):
        for fn in os.listdir(MODEL_DIR):
            if fn.lower().endswith(".joblib") or fn.lower().endswith(".pkl"):
                model_files.append(os.path.join(MODEL_DIR, fn))
    if not model_files:
        print(f"No models found in {MODEL_DIR}. Place .joblib models there (see README).")
        sys.exit(1)

    all_metrics = []
    for mpath in model_files:
        print("Evaluating", mpath)
        m = evaluate_model(mpath, X, y)
        all_metrics.append(m)

    out_path = os.path.join(RESULTS_DIR, "metrics.md")
    save_results(all_metrics, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
