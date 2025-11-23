# app_dual_model_pt.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

st.set_page_config(page_title="Demo: IMC + Modelo Completo", layout="wide")

# ------- CONFIG ----------
MODELS_DIR = Path("models")
DATA_PATH = Path("/mnt/data/Obesity.csv") if Path("/mnt/data/Obesity.csv").exists() else Path("Obesity.csv")

# ------- UTILS -------------
def find_models(models_dir):
    models_dir = Path(models_dir)
    files = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
    bmi_path, multi_path = None, None
    for f in files:
        lf = f.lower()
        # heurísticas
        if "bmi" in lf and bmi_path is None:
            bmi_path = models_dir / f
        if ("multi" in lf or "no_bmi" in lf or "class_weight" in lf or "pipeline_class_weight" in lf) and multi_path is None:
            multi_path = models_dir / f
    # fallback simples
    if bmi_path is None:
        for f in files:
            if "bmi" in f.lower():
                bmi_path = models_dir / f
                break
    if multi_path is None:
        for f in files:
            if bmi_path and f == bmi_path.name:
                continue
            if "pipeline" in f.lower() or "multi" in f.lower() or "class_weight" in f.lower():
                multi_path = models_dir / f
                break
    return bmi_path, multi_path, files

def to_float(v, default):
    if v is None:
        return default
    if isinstance(v, str):
        v = v.replace(",", ".")
    try:
        return float(v)
    except:
        return default

def safe_load(path):
    if not path:
        return None
    try:
        m = joblib.load(path)
        # if dict wrap, try to extract usual keys
        if isinstance(m, dict):
            for k in ("model","pipeline","estimator","clf","clf_cw"):
                if k in m:
                    return m[k]
        return m
    except Exception as e:
        st.sidebar.error(f"Erro carregando {path.name}: {e}")
        return None

def prepare_bmi_df_for_model(model, bmi_val):
    # tenta adaptar para o que o modelo espera
    if model is None:
        return None
    if isinstance(model, dict):
        # fallback: try first value
        model = list(model.values())[0] if model else model
    cols = None
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
    if not cols:
        # simples: criar coluna BMI
        return pd.DataFrame([[bmi_val]], columns=["BMI"])
    # se modelo espera várias colunas, preenche BMI-like e zeros para o resto
    row = {}
    for c in cols:
        row[c] = bmi_val if 'bmi' in c.lower() else 0
    X = pd.DataFrame([row])
    # se só havia uma coluna confidencial, renomeia para BMI
    if len(X.columns) == 1:
        X.columns = ["BMI"]
    return X

def adapt_for_model(model, X):
    if model is None:
        return None
    if isinstance(model, dict):
        model = list(model.values())[0]
    if hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)
        for c in names:
            if c not in X.columns:
                X[c] = 0
        return X[names]
    return X

# ------- UI styling ----------
st.markdown(
    """
    <style>
    .stApp { background-color: #0f1720; color: #ffffff; }
    .big-title { font-size:36px; font-weight:700; }
    .card { background-color:#0b1220; padding:16px; border-radius:8px; margin-bottom:12px; }
    .muted { color:#9aa4b2; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Demo: Previsão de Obesidade — IMC (rápido) + Modelo Completo (opcional)")
st.write("Mostre ao usuário um resultado rápido (apenas Peso/Altura) e uma análise completa quando ele fornecer mais informações.")
st.caption(f"Dataset local (para referências): `{DATA_PATH}`")

# ------- carregar modelos -------------
if not MODELS_DIR.exists():
    st.sidebar.error("Pasta `models/` não encontrada. Coloque seus .joblib lá.")
else:
    bmi_path, multi_path, model_files = find_models(MODELS_DIR)
    st.sidebar.header("Modelos detectados")
    st.sidebar.write([f for f in model_files])
    if bmi_path:
        st.sidebar.success(f"Modelo BMI detectado: {bmi_path.name}")
    else:
        st.sidebar.warning("Modelo BMI NÃO detectado.")
    if multi_path:
        st.sidebar.success(f"Modelo completo detectado: {multi_path.name}")
    else:
        st.sidebar.warning("Modelo completo NÃO detectado.")

bmi_model = safe_load(bmi_path) if bmi_path else None
multi_model = safe_load(multi_path) if multi_path else None

# ------- Dados de exemplo para defaults ----------
sample = {}
if DATA_PATH.exists():
    try:
        df_data = pd.read_csv(DATA_PATH)
        if not df_data.empty:
            sample = df_data.iloc[0].to_dict()
    except Exception:
        sample = {}

# ------- 1) Resultado rápido (IMC) ----------
st.subheader("1) Resultado rápido — apenas Peso e Altura (IMC)")
col1, col2, col3 = st.columns([1,1,1.2])
with col1:
    weight_default = to_float(sample.get('Weight', 70), 70.0)
    weight = st.number_input("Peso (kg)", min_value=1.0, max_value=400.0, value=weight_default, step=0.1, format="%.1f")
with col2:
    height_default = to_float(sample.get('Height', 1.70), 1.70)
    height = st.number_input("Altura (m)", min_value=0.5, max_value=2.5, value=height_default, step=0.01, format="%.2f")
with col3:
    bmi_val = weight / (height**2) if height > 0 else 0.0
    st.metric("IMC (BMI)", f"{bmi_val:.2f}")

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.write("**Resultado Rápido**")
bmi_input_df = prepare_bmi_df_for_model(bmi_model, bmi_val)
pred_bmi, prob_bmi = None, None
if bmi_model is not None and bmi_input_df is not None:
    try:
        pred_bmi = bmi_model.predict(bmi_input_df)[0]
        st.write(f"Predição (IMC): **{pred_bmi}**")
    except Exception:
        st.warning("Modelo BMI não aceitou entrada (verifique pipeline).")
    try:
        prob_arr = bmi_model.predict_proba(bmi_input_df)
        prob_bmi = prob_arr.max(axis=1)[0]
        st.write(f"Confiança (max prob): {prob_bmi:.3f}")
    except Exception:
        pass
else:
    st.info("Modelo BMI não disponível — carregue um pipeline em models/ (ex: bmi_pipeline.joblib)")
st.markdown("</div>", unsafe_allow_html=True)

st.write("---")

# ------- 2) Formulário completo (em português, agrupado) ----------
st.subheader("2) Resultado completo (opcional) — responda estas perguntas para uma análise mais rica")
with st.expander("Mostrar formulário completo"):
    with st.form("form_completo"):
        st.markdown("**Informações básicas**")
        gender = st.selectbox("Gênero", options=["Female","Male"], index=(0 if sample.get('Gender','Female')=='Female' else 1))
        age = st.number_input("Idade (anos)", min_value=1, max_value=120, value=int(sample.get('Age',25) if sample else 25))

        st.markdown("**Hábitos alimentares**")
        family_history = st.selectbox("Histórico familiar de obesidade?", options=["yes","no"], index=0 if sample.get('family_history','yes')=='yes' else 1)
        favc = st.selectbox("Consumo frequente de alimentos calóricos?", options=["yes","no"], index=0 if sample.get('FAVC','yes')=='yes' else 1)
        fcvc = st.slider("Consumo de verduras (1=baixo → 5=alto)", 1, 5, value=int(sample.get('FCVC',2) if sample else 2))
        ncp = st.slider("Número de refeições principais por dia", 1, 6, value=int(sample.get('NCP',3) if sample else 3))
        caec = st.selectbox("Come entre refeições?", options=["no","Sometimes","Frequently","Always"], index=0)

        st.markdown("**Estilo de vida**")
        smoke = st.selectbox("Fuma?", options=["no","yes"], index=0)
        ch2o = st.slider("Consumo diário de água (1-5)", 1, 5, value=int(sample.get('CH2O',2) if sample else 2))
        scc = st.selectbox("Monitora calorias regularmente?", options=["no","yes"], index=0)
        faf = st.slider("Frequência de atividade física (1-5)", 1, 5, value=int(sample.get('FAF',2) if sample else 2))
        tue = st.number_input("Horas por dia em telas/dispositivos (TUE)", min_value=0, max_value=24, value=int(sample.get('TUE',2) if sample else 2))
        calc = st.selectbox("Consumo de álcool", options=["no","Sometimes","Frequently","Always"], index=0)
        mtrans = st.selectbox("Transporte habitual", options=["Public_Transportation","Walking","Automobile","Motorbike","Bike"], index=0)

        submitted = st.form_submit_button("Analisar (modelo completo)")

    if submitted:
        full_row = {
            "Gender": gender,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "family_history": family_history,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans
        }
        X_full = pd.DataFrame([full_row])
        X_multi_for_model = adapt_for_model(multi_model, X_full.copy()) if multi_model is not None else None

        pred_multi, prob_multi = None, None
        if multi_model is not None:
            try:
                pred_multi = multi_model.predict(X_multi_for_model)[0]
                try:
                    prob_multi = multi_model.predict_proba(X_multi_for_model).max(axis=1)[0]
                except Exception:
                    prob_multi = None
            except Exception as e:
                st.error(f"Erro ao predizer com o modelo completo: {e}")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Resultado completo**")
        if pred_multi:
            st.write(f"Predição (modelo completo): **{pred_multi}**")
            if prob_multi is not None:
                st.write(f"Confiança (max prob): {prob_multi:.3f}")
        else:
            st.warning("Modelo completo não disponível ou erro na predição.")
        st.markdown("</div>", unsafe_allow_html=True)

        # regra híbrida (decisão final)
        final, reason = None, ""
        if pred_bmi is None and pred_multi is None:
            st.error("Nenhum modelo disponível para decisão.")
        elif pred_bmi is not None and pred_multi is None:
            final = pred_bmi; reason = "Apenas IMC disponível"
        elif pred_bmi is None and pred_multi is not None:
            final = pred_multi; reason = "Apenas modelo completo disponível"
        else:
            if pred_bmi == pred_multi:
                final = pred_bmi; reason = "Modelos concordam"
            else:
                if prob_bmi is None or prob_multi is None:
                    final = pred_bmi; reason = "Desempate: preferido IMC (falta probabilidades)"
                else:
                    delta = 0.10
                    if abs(prob_bmi - prob_multi) >= delta:
                        final = pred_bmi if prob_bmi > prob_multi else pred_multi
                        reason = f"Desempate por confiança (IMC {prob_bmi:.2f} vs Multi {prob_multi:.2f})"
                    else:
                        final = pred_bmi
                        reason = "Probabilidades semelhantes — fallback para IMC"

        if final:
            st.success(f"Decisão final: **{final}** — {reason}")

        # salvar discordância
        if pred_bmi != pred_multi:
            outp = X_full.copy()
            outp['pred_bmi'] = pred_bmi
            outp['prob_bmi'] = prob_bmi
            outp['pred_multi'] = pred_multi
            outp['prob_multi'] = prob_multi
            outp['final'] = final
            outp['reason'] = reason
            outpath = MODELS_DIR / "discordance_examples.csv"
            if outpath.exists():
                try:
                    outp.to_csv(outpath, mode='a', header=False, index=False)
                except Exception:
                    pass
            else:
                outp.to_csv(outpath, index=False)
            st.info(f"Exemplo salvo em {outpath}")

st.write("---")
st.subheader("Arquivos / artefatos")
if MODELS_DIR.exists():
    files = sorted([p.name for p in MODELS_DIR.glob("*")])
    st.write(files)
if DATA_PATH.exists():
    st.markdown(f"Dataset local: `{DATA_PATH}`")
