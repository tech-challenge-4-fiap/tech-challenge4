# compare_bmi_vs_multi.py
import os, json, argparse
import joblib
import pandas as pd, numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='models')
    args = parser.parse_args()

    out = args.outdir
    test_path = os.path.join(out, 'test_set.csv')
    if not os.path.exists(test_path):
        raise SystemExit("test_set.csv não encontrado em models/. Rode train_two_models.py primeiro.")
    df_test = pd.read_csv(test_path)
    # load models
    m_multi_path = os.path.join(out, 'multi_no_bmi.joblib')
    m_bmi_path = os.path.join(out, 'bmi_only.joblib')
    if not os.path.exists(m_multi_path) or not os.path.exists(m_bmi_path):
        raise SystemExit("Modelos não encontrados. Verifique models/*.joblib")
    m_multi = joblib.load(m_multi_path)
    m_bmi = joblib.load(m_bmi_path)
    # detect target col in test df (we saved full original)
    target = None
    for c in df_test.columns:
        if 'obes' in c.lower():
            target = c
            break
    if target is None:
        target = df_test.columns[-1]
    y_true = df_test[target].astype(str).values

    # build X matrices for each model: infer required columns using feature_names_in_ or by pipeline
    def build_input_for_model(model, df):
        # If pipeline with ColumnTransformer named 'pre' etc, try to get input names via fit info
        # Simpler approach: attempt to call predict on a DataFrame that contains all df columns (pipeline will pick necessary)
        return df.drop(columns=[target])

    X_all = build_input_for_model(m_multi, df_test)
    # For BMI-only, build DataFrame with BMI column only if pipeline expects it
    if hasattr(m_bmi, 'feature_names_in_'):
        bmi_cols = [c for c in m_bmi.feature_names_in_ if 'bmi' in c.lower()]
        if bmi_cols:
            X_bmi = df_test[bmi_cols]
        else:
            X_bmi = df_test[['BMI']]
    else:
        if 'BMI' in df_test.columns:
            X_bmi = df_test[['BMI']]
        else:
            raise SystemExit("Não encontrei coluna BMI no test_set.csv")

    # predict
    pred_multi = m_multi.predict(X_all)
    prob_multi = (m_multi.predict_proba(X_all) if hasattr(m_multi, 'predict_proba') else None)
    pred_bmi = m_bmi.predict(X_bmi)
    prob_bmi = (m_bmi.predict_proba(X_bmi) if hasattr(m_bmi, 'predict_proba') else None)

    # per-model metrics
    def summarize(y_true, y_pred, probs, name):
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
        return {
            'report': report,
            'confusion_matrix': cm,
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro'))
        }

    res_bmi = summarize(y_true, pred_bmi, prob_bmi, 'bmi_only')
    res_multi = summarize(y_true, pred_multi, prob_multi, 'multi_no_bmi')

    # build comparison CSV
    rows = []
    for i in range(len(y_true)):
        r = {
            'index': i,
            'y_true': y_true[i],
            'pred_bmi': str(pred_bmi[i]),
            'pred_multi': str(pred_multi[i])
        }
        if prob_bmi is not None:
            r['prob_bmi_max'] = float(max(prob_bmi[i]))
        if prob_multi is not None:
            r['prob_multi_max'] = float(max(prob_multi[i]))
        # keep group col if exists
        for c in df_test.columns:
            if c.lower() in ('gender','sex','genero'):
                r[c] = df_test.iloc[i][c]
        rows.append(r)
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(out, 'compare_predictions.csv'), index=False)

    summary = {'bmi_only': res_bmi, 'multi_no_bmi': res_multi}
    with open(os.path.join(out, 'compare_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== SUMMARY ===")
    print("BMI-only: accuracy", res_bmi['accuracy'], "f1_macro", res_bmi['f1_macro'])
    print("Multi(no BMI): accuracy", res_multi['accuracy'], "f1_macro", res_multi['f1_macro'])
    print("Files saved to", out, ": compare_predictions.csv and compare_summary.json")

if __name__ == '__main__':
    main()
