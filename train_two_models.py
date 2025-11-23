# train_two_models.py
import os, json, argparse
import joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

def load_df(path):
    df = pd.read_csv(path)
    # criar BMI se faltar
    if 'BMI' not in df.columns and 'Weight' in df.columns and 'Height' in df.columns:
        df['BMI'] = df['Weight'] / (df['Height']**2)
    return df

def get_target(df):
    if 'Obesity' in df.columns:
        return 'Obesity'
    # tenta variações
    for c in df.columns:
        if 'obes' in c.lower():
            return c
    return df.columns[-1]

def detect_cols(df, target):
    feat = [c for c in df.columns if c != target]
    num = [c for c in feat if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in feat if c not in num]
    return num, cat

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='mean'))])
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', ohe)])
    pre = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ], remainder='drop')
    return pre

def train_and_save(X_train, X_test, y_train, y_test, name, num_cols, cat_cols, outdir):
    pre = build_preprocessor(num_cols, cat_cols)
    pipe = Pipeline([('pre', pre), ('clf', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))])
    param_grid = {'clf__n_estimators': [100], 'clf__max_depth': [5, None]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    probs = (best.predict_proba(X_test) if hasattr(best, "predict_proba") else None)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    res = {
        'name': name,
        'best_params': gs.best_params_,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'report': report,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    os.makedirs(outdir, exist_ok=True)
    model_path = os.path.join(outdir, f'{name}.joblib')
    joblib.dump(best, model_path)
    with open(os.path.join(outdir, f'results_{name}.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    return best, res, probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--outdir', default='models')
    args = parser.parse_args()

    df = load_df(args.data)
    target = get_target(df)
    print("Target:", target)
    # Remover colunas não-informativas óbvias (IDs) - ajustar se tiver
    # decidir features a usar para 'multi_no_bmi': todas exceto BMI e target
    all_feats = [c for c in df.columns if c != target]
    feats_multi = [c for c in all_feats if c != 'BMI']  # EXPLÍCITO: remove BMI
    feats_bmi = ['BMI']

    # construir X,y para split idêntico
    X_all = df[feats_multi].copy()
    y = df[target].astype(str).copy()

    # stratified split
    X_train_m, X_test_m, y_train, y_test = train_test_split(X_all, y, test_size=0.20, stratify=y, random_state=42)
    # Para BMI-only: construir X from original df using same test indices
    test_idx = X_test_m.index
    X_test_bmi = df.loc[test_idx, feats_bmi].copy()
    X_train_bmi = df.loc[X_train_m.index, feats_bmi].copy()

    # detect numerics/categoricals for multi
    num_multi = [c for c in feats_multi if pd.api.types.is_numeric_dtype(df[c])]
    cat_multi = [c for c in feats_multi if c not in num_multi]

    # for bmi-only
    num_bmi = ['BMI']
    cat_bmi = []

    os.makedirs(args.outdir, exist_ok=True)
    # save test set (so compare script can reuse)
    test_save = df.loc[test_idx].copy()
    test_save.to_csv(os.path.join(args.outdir, 'test_set.csv'), index=False)
    print("Saved test_set.csv with", len(test_save), "rows to", args.outdir)

    print("Training multi_no_bmi with features:", feats_multi)
    model_multi, res_multi, probs_multi = train_and_save(X_train_m, X_test_m, y_train, y_test, 'multi_no_bmi', num_multi, cat_multi, args.outdir)
    print("multi_no_bmi done - f1_macro:", res_multi['f1_macro'])

    # train bmi-only
    print("Training bmi_only (BMI only)")
    model_bmi, res_bmi, probs_bmi = train_and_save(X_train_bmi, X_test_bmi, y_train, y_test, 'bmi_only', num_bmi, cat_bmi, args.outdir)
    print("bmi_only done - f1_macro:", res_bmi['f1_macro'])

    # save a short compare summary
    compare_summary = {'multi_no_bmi': res_multi, 'bmi_only': res_bmi}
    with open(os.path.join(args.outdir, 'results_compare.json'), 'w', encoding='utf-8') as f:
        json.dump(compare_summary, f, indent=2, ensure_ascii=False)
    print("Saved compare summary to", os.path.join(args.outdir, 'results_compare.json'))

if __name__ == '__main__':
    main()
