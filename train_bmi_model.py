# train_bmi_model.py
import argparse, os, json
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--outdir", default="models")
    return p.parse_args()

def load_and_prep(path):
    df = pd.read_csv(path)
    # detect target
    if 'Obesity' in df.columns:
        target = 'Obesity'
    elif 'Obesity_level' in df.columns:
        target = 'Obesity_level'
    else:
        target = df.columns[-1]
    # compute BMI & drop raw cols if present
    if 'Weight' in df.columns and 'Height' in df.columns:
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        # keep weight/height only if you want; to prevent multicollinearity we drop them
        df = df.drop(columns=['Weight','Height'])
    # basic mapping of yes/no to numeric if any
    for c in df.select_dtypes(include=['object']).columns:
        vals = df[c].dropna().unique()
        low = [str(x).lower() for x in vals]
        if set(low) <= {'yes','no'}:
            df[c] = df[c].map(lambda x: 1 if str(x).lower()=='yes' else 0)
    return df, target

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df, target = load_and_prep(args.data)
    y = df[target].astype(str)
    X = df[['BMI']].copy()  # only BMI feature

    # optional group col for fairness checks
    group_col = None
    for c in df.columns:
        if c.lower() in ('gender','genero','sex'):
            group_col = c
            break

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # pipeline: scaler + classifier (we'll evaluate LogisticRegression and RandomForest)
    pipe_lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=200, class_weight='balanced', random_state=args.random_state))])
    pipe_rf = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_jobs=-1, random_state=args.random_state, class_weight='balanced'))])

    # quick CV to decide model (StratifiedKFold)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    # Grid search over model hyperparams (small search)
    param_grid = [
        {'clf': [LogisticRegression(max_iter=200, class_weight='balanced', random_state=args.random_state)], 'clf__C': [0.01, 0.1, 1, 10]},
        {'clf': [RandomForestClassifier(n_jobs=-1, class_weight='balanced', random_state=args.random_state)], 'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [None, 5, 10]}
    ]
    from sklearn.model_selection import GridSearchCV
    pipe_base = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])  # placeholder
    gs = GridSearchCV(pipe_base, param_grid, cv=skf, scoring='f1_macro', n_jobs=-1)
    gs.fit(X_train, y_train)
    print("Best CV params:", gs.best_params_, "best_score (f1_macro):", gs.best_score_)

    best_pipe = gs.best_estimator_
    # calibrate probabilities (optional for better probabilities)
    calibrated = CalibratedClassifierCV(best_pipe, cv='prefit')  # fits calibrator using validation folds internally
    calibrated.fit(X_train, y_train)

    # eval on test
    y_pred = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test) if hasattr(calibrated, 'predict_proba') else None
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("Test accuracy:", acc)
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", cm)

    # per-group metrics (if group_col present on original df)
    per_group = {}
    if group_col:
        X_test_full = df.loc[idx_test]
        groups = X_test_full[group_col].unique()
        for g in groups:
            mask = X_test_full[group_col] == g
            if mask.sum() < 3: continue
            y_t = y_test[mask]
            y_p = pd.Series(y_pred, index=y_test.index)[mask]
            per_group[str(g)] = classification_report(y_t, y_p, output_dict=True)

    # save artifacts
    joblib.dump({'model': calibrated, 'feature_cols': ['BMI']}, os.path.join(args.outdir, 'bmi_pipeline.joblib'))
    results = {'accuracy': acc, 'report': report, 'confusion_matrix': cm, 'per_group': per_group, 'best_params': str(gs.best_params_)}
    with open(os.path.join(args.outdir, 'results_bmi.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("Saved model to:", os.path.join(args.outdir, 'bmi_pipeline.joblib'))
    print("Saved results to:", os.path.join(args.outdir, 'results_bmi.json'))

if __name__ == "__main__":
    main()
