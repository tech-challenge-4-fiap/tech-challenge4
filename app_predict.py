# app_predict_robust.py
import streamlit as st, joblib, os, numpy as np, pandas as pd

st.set_page_config(page_title="Predictor de Obesidade (BMI-only)")
st.title("Predictor de Obesidade (BMI-only)")

DEFAULT_MODEL_PATH = "models/bmi_pipeline.joblib"

def load_model(path):
    if not os.path.exists(path):
        st.error(f"Arquivo não encontrado: {path}")
        return None
    try:
        m = joblib.load(path)
        # if dict with 'model'
        if isinstance(m, dict):
            for k in ('model','clf','pipeline','estimator'):
                if k in m:
                    return m[k]
            # fallback: try first value
            return list(m.values())[0]
        return m
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

if 'mdl' not in st.session_state:
    st.session_state.mdl = None
    if os.path.exists(DEFAULT_MODEL_PATH):
        st.session_state.mdl = load_model(DEFAULT_MODEL_PATH)
        if st.session_state.mdl:
            st.success(f"Modelo carregado automaticamente: {DEFAULT_MODEL_PATH}")

model_path = st.text_input("Caminho do modelo", DEFAULT_MODEL_PATH)
if st.button("Carregar modelo"):
    st.session_state.mdl = load_model(model_path)
    if st.session_state.mdl:
        st.success("Modelo carregado")

peso = st.number_input("Peso (kg)", 10.0, 500.0, 70.0, step=0.1)
altura = st.number_input("Altura (m)", 0.5, 2.5, 1.70, step=0.01)

if st.button("Prever"):
    bmi = peso / (altura**2)
    st.write(f"IMC calculado: {bmi:.2f}")
    mdl = st.session_state.mdl
    if mdl is None:
        st.error("Modelo não carregado.")
    else:
        try:
            # caso 1: pipeline or estimator that accepts 2D array with one column
            X_try = np.array([[bmi]])
            try:
                pred = mdl.predict(X_try)[0]
                st.success(f"Predição: {pred}")
                if hasattr(mdl, "predict_proba"):
                    st.write("Probs:", mdl.predict_proba(X_try).round(3).tolist())
            except Exception:
                # caso 2: estimator expecting many features. Try to build DataFrame from feature_names_in_
                if hasattr(mdl, 'feature_names_in_'):
                    feat_names = list(mdl.feature_names_in_)
                    # build defaults (zeros) and set BMI
                    defaults = {f: 0 for f in feat_names}
                    # find BMI-like column
                    possible_bmi_cols = [c for c in feat_names if 'bmi' in c.lower() or 'imc' in c.lower()]
                    if possible_bmi_cols:
                        defaults[possible_bmi_cols[0]] = bmi
                    else:
                        # try common name
                        if 'BMI' in feat_names:
                            defaults['BMI'] = bmi
                    X_df = pd.DataFrame([defaults], columns=feat_names)
                    pred = mdl.predict(X_df)[0]
                    st.success(f"Predição (com preenchimento de features): {pred}")
                    if hasattr(mdl, "predict_proba"):
                        st.write("Probs:", mdl.predict_proba(X_df).round(3).tolist())
                else:
                    st.error("Modelo espera múltiplas features e não expõe 'feature_names_in_'. Nesse caso é melhor recriar um pipeline que aceite somente BMI.")
        except Exception as e:
            st.error(f"Erro inesperado: {e}")

st.markdown("---")
st.write("Diagnóstico:")
if 'mdl' in st.session_state and st.session_state.mdl is not None:
    st.write("Tipo do objeto:", type(st.session_state.mdl))
    st.write("n_features_in_:", getattr(st.session_state.mdl, "n_features_in_", None))
    st.write("feature_names_in_ (first 30):", list(getattr(st.session_state.mdl, "feature_names_in_", []) )[:30])
else:
    st.write("Nenhum modelo carregado.")
