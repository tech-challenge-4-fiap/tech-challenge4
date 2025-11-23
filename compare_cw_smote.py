#!/usr/bin/env python3
"""
compare_cw_smote.py
Comparativo: RandomForest (class_weight='balanced') vs RandomForest + SMOTE (ou fallback).
Gera CV summaries, avaliação final em test set, métricas por grupo (Gender) e salva modelos/resultados.

Usage:
    python compare_cw_smote.py --data /mnt/data/Obesity.csv --folds 5 --trees 100 --outdir results
"""
import argparse
import os
import json
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
import joblib
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Caminho para CSV (ex: /mnt/data/Obesity.csv)")
    p.add_argument("--folds", type=int, default=5, help="Folds CV (default: 5)")
    p.add_argument("--trees", type=int, default=100, help="n_estimators RandomForest (default: 100)")
    p.add_argument("--outdir", default="results", help="Diretório de saída (default: results)")
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()

def load_and_prep(path):
    df = pd.read_csv(path)
    # Detect target column
    if 'Obesity' in df.columns:
        target_col = 'Obesity'
    elif 'Obesity_level' in df.columns:
        target_col = 'Obesity_level'
    else:
        target_col = df.columns[-1]
    # create BMI if Height & Weight exist
    if 'Weight' in df.columns and 'Height' in df.columns:
        df['BMI'] = df['Weight'] / (df['Height']**2)
        df = df.drop(columns=['Weight','Height'])
    # Map yes/no -> 1/0 in object cols when applicable
    for col in df.select_dtypes(include=['object']).columns.tolist():
        uniques = df[col].dropna().unique().tolist()
        lower_uniques = [str(x).lower() for x in uniques]
        if set(lower_uniques) <= {'yes','no'}:
            df[col] = df[col].map(lambda x: 1 if str(x).lower()=='yes' else 0)
    return df, target_col

def identify_group_col(X):
    for c in X.columns:
        if c.lower() in ('gender','genero','sex'):
            return c
    return None

# Substitua sua função build_preprocessor por esta versão
from sklearn.preprocessing import OneHotEncoder

def _make_onehot():
    """
    Create OneHotEncoder that returns dense arrays in a way compatible with both
    older and newer scikit-learn versions.
    """
    try:
        # scikit-learn >= ~1.2 uses sparse_output
        return OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
    except TypeError:
        # older scikit-learn uses sparse
        return OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)

def build_preprocessor(X):
    """
    X: a DataFrame (used only to detect feature types)
    returns: ColumnTransformer, numeric_feats, cat_feats
    """
    numeric_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X.select_dtypes(include=['object','category']).columns.tolist()

    num_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([('onehot', _make_onehot())]) if len(cat_feats) > 0 else None

    transformers = [('num', num_pipe, numeric_feats)]
    if cat_pipe:
        transformers.append(('cat', cat_pipe, cat_feats))

    preproc = ColumnTransformer(transformers=transformers, remainder='drop')
    return preproc, numeric_feats, cat_feats


def compute_per_group_metrics(y_true, y_pred, X_test, group_col):
    per_group = {}
    if group_col and group_col in X_test.columns:
        for g in X_test[group_col].unique():
            idx = X_test[group_col] == g
            if idx.sum() < 3:
                continue
            y_t = y_true[idx]
            y_p = y_pred[idx]
            per_group[str(g)] = {
                'accuracy': accuracy_score(y_t, y_p),
                'precision_macro': precision_score(y_t, y_p, average='macro', zero_division=0),
                'recall_macro': recall_score(y_t, y_p, average='macro', zero_division=0),
                'f1_macro': f1_score(y_t, y_p, average='macro', zero_division=0),
                'support': int(idx.sum())
            }
    return per_group

def manual_cv(X_train, y_train, folds, n_trees, use_smote, random_state):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    metrics = defaultdict(list)
    # detect imblearn
    try:
        from imblearn.over_sampling import SMOTE  # noqa: F401
        smote_available = True
    except Exception:
        smote_available = False

    preproc, _, _ = build_preprocessor(X_train)
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        X_tr_t = preproc.fit_transform(X_tr)
        X_val_t = preproc.transform(X_val)
        # oversample if requested
        if use_smote and smote_available:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=random_state, k_neighbors=3)
            X_res, y_res = sm.fit_resample(X_tr_t, y_tr)
            clf = RandomForestClassifier(n_estimators=n_trees, random_state=random_state, n_jobs=1)
        elif use_smote and not smote_available:
            # fallback random oversampling in transformed space
            tr_df = pd.DataFrame(X_tr_t)
            tr_df['target'] = y_tr.values
            maxc = tr_df['target'].value_counts().max()
            up = []
            for cls, grp in tr_df.groupby('target'):
                if len(grp) < maxc:
                    up.append(resample(grp, replace=True, n_samples=maxc, random_state=random_state))
                else:
                    up.append(grp)
            up_df = pd.concat(up).sample(frac=1, random_state=random_state)
            X_res = up_df.drop(columns=['target']).values
            y_res = up_df['target'].values
            clf = RandomForestClassifier(n_estimators=n_trees, random_state=random_state, n_jobs=1)
        else:
            X_res, y_res = X_tr_t, y_tr.values
            clf = RandomForestClassifier(n_estimators=n_trees, class_weight='balanced', random_state=random_state, n_jobs=1)

        clf.fit(X_res, y_res)
        y_pred = clf.predict(X_val_t)
        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['precision_macro'].append(precision_score(y_val, y_pred, average='macro', zero_division=0))
        metrics['recall_macro'].append(recall_score(y_val, y_pred, average='macro', zero_division=0))
        metrics['f1_macro'].append(f1_score(y_val, y_pred, average='macro', zero_division=0))
    # summarize
    summary = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in metrics.items()}
    return summary, smote_available

def train_final_and_eval(X_train, y_train, X_test, y_test, n_trees, use_smote, random_state):
    preproc, _, _ = build_preprocessor(X_train)
    X_tr_t = preproc.fit_transform(X_train)
    X_test_t = preproc.transform(X_test)
    # check SMOTE availability
    try:
        from imblearn.over_sampling import SMOTE
        smote_available = True
    except Exception:
        smote_available = False
    if use_smote and smote_available:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state, k_neighbors=3)
        X_res, y_res = sm.fit_resample(X_tr_t, y_train)
        clf = RandomForestClassifier(n_estimators=n_trees, random_state=random_state, n_jobs=1)
    elif use_smote and not smote_available:
        tr_df = pd.DataFrame(X_tr_t); tr_df['target'] = y_train.values
        maxc = tr_df['target'].value_counts().max()
        ups = []
        for cls, grp in tr_df.groupby('target'):
            ups.append(resample(grp, replace=True, n_samples=maxc, random_state=random_state) if len(grp)<maxc else grp)
        up_df = pd.concat(ups).sample(frac=1, random_state=random_state)
        X_res = up_df.drop(columns=['target']).values
        y_res = up_df['target'].values
        clf = RandomForestClassifier(n_estimators=n_trees, random_state=random_state, n_jobs=1)
    else:
        X_res, y_res = X_tr_t, y_train.values
        clf = RandomForestClassifier(n_estimators=n_trees, class_weight='balanced', random_state=random_state, n_jobs=1)

    clf.fit(X_res, y_res)
    y_pred = clf.predict(X_test_t)
    report = classification_report(y_test, y_pred, output_dict=True)
    confmat = confusion_matrix(y_test, y_pred).tolist()
    return report, confmat, clf, preproc

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    df, target_col = load_and_prep(args.data)
    print("Data loaded:", df.shape, "Target:", target_col)
    y = df[target_col].astype(str)
    X = df.drop(columns=[target_col])
    group_col = identify_group_col(X)
    print("Group column used for per-group metrics:", group_col)
    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state, stratify=y)
    print("Train/test sizes:", X_train.shape, X_test.shape)

    # Manual CV: class_weight
    print("Running CV (class_weight='balanced')...")
    cw_summary, _ = manual_cv(X_train, y_train, folds=args.folds, n_trees=args.trees, use_smote=False, random_state=args.random_state)

    print("Running CV (SMOTE/oversample)...")
    sm_summary, smote_available = manual_cv(X_train, y_train, folds=args.folds, n_trees=args.trees, use_smote=True, random_state=args.random_state)

    # Final train+eval on test
    print("Training final (class_weight) and evaluating on test set...")
    rep_cw, cm_cw, clf_cw, pre_cw = train_final_and_eval(X_train, y_train, X_test, y_test, n_trees=args.trees, use_smote=False, random_state=args.random_state)
    per_group_cw = compute_per_group_metrics(y_test, clf_cw.predict(pre_cw.transform(X_test)), X_test, group_col)

    print("Training final (SMOTE/oversample) and evaluating on test set...")
    rep_sm, cm_sm, clf_sm, pre_sm = train_final_and_eval(X_train, y_train, X_test, y_test, n_trees=args.trees, use_smote=True, random_state=args.random_state)
    per_group_sm = compute_per_group_metrics(y_test, clf_sm.predict(pre_sm.transform(X_test)), X_test, group_col)

    # Save artifacts
    joblib.dump({'clf_cw': clf_cw, 'pre_cw': pre_cw}, os.path.join('models', 'pipeline_class_weight.joblib'))
    joblib.dump({'clf_sm': clf_sm, 'pre_sm': pre_sm}, os.path.join('models', 'pipeline_smote_or_oversample.joblib'))
    results = {
        'cv_class_weight': cw_summary,
        'cv_smote': sm_summary,
        'test_class_weight_report': rep_cw,
        'test_class_weight_confusion': cm_cw,
        'test_smote_report': rep_sm,
        'test_smote_confusion': cm_sm,
        'per_group_cw': per_group_cw,
        'per_group_sm': per_group_sm,
        'smote_available': smote_available,
        'target_column': target_col,
        'group_column': group_col
    }
    # Save JSON + joblib
    with open(os.path.join(args.outdir, "detailed_reports.json"), "w") as f:
        json.dump(results, f, indent=2)
    joblib.dump(results, os.path.join(args.outdir, "detailed_reports.joblib"))

    # Create a CSV summary table (method, accuracy_test, f1_macro_test)
    def extract_summary(rep):
        acc = rep.get('accuracy', None)
        f1_macro = rep.get('macro avg', {}).get('f1-score', None)
        return acc, f1_macro

    acc_cw, f1_cw = extract_summary(rep_cw)
    acc_sm, f1_sm = extract_summary(rep_sm)
    summary_df = pd.DataFrame([
        {'method': 'class_weight', 'accuracy_test': acc_cw, 'f1_macro_test': f1_cw},
        {'method': 'smote_or_oversample', 'accuracy_test': acc_sm, 'f1_macro_test': f1_sm}
    ])
    summary_df.to_csv(os.path.join(args.outdir, "compare_summary.csv"), index=False)

    print("\nDone. Artifacts saved:")
    print(" - models/pipeline_class_weight.joblib")
    print(" - models/pipeline_smote_or_oversample.joblib")
    print(" - {}/detailed_reports.json".format(args.outdir))
    print(" - {}/compare_summary.csv".format(args.outdir))
    print("\nResumo (test):")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
