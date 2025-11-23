# app_dual_model_pt_full.py
"""
Demo Streamlit: IMC (rápido) + Modelo Completo (opcional)
- Mapeamento Sim/Não -> yes/no para compatibilidade com pipelines treinados
- Salva feedback do usuário em models/feedback.csv
- Salva exemplos de discordância em models/discordance_examples.csv
- Admin view para inspecionar artifacts
Dataset local esperado: /mnt/data/Obesity.csv
Run: streamlit run app_dual_model_pt_full.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import csv
import time
from pathlib import Path
from io import BytesIO

st.set_page_config(page_title="IMC + Modelo Completo (PT) — Demo", layout="wide")

# ---------------- CONFIG ----------------
MODELS_DIR = Path("models")
DATA_PATH = Path("/mnt/data/Obesity.csv") if Path("/mnt/data/Obesity.csv").exists() else Path("Obesity.csv")
DISCORDANCE_FILE = MODELS_DIR / "discordance_examples.csv"
FEEDBACK_FILE = MODELS_DIR / "feedback.csv"

# ---------------- UTIL ----------------
def ensure_models_dir():
    if not MODELS_DIR.exists():
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

def find_models(models_dir):
    models_dir = Path(models_dir)
    files = [f for f in os.listdir(models_dir) if f.endswith(".joblib")] if models_dir.exists() else []
    bmi_path, multi_path = None, None
    for f in files:
        lf = f.lower()
        if "bmi" in lf and bmi_path is None:
            bmi_path = models_dir / f
        if ("multi" in lf or "no_bmi" in lf or "class_weight" in lf or "pipeline_class_weight" in lf) and multi_path is None:
            multi_path = models_dir / f
    # fallback heuristics
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

@st.cache_resource
def safe_load_model(path):
    if not path:
        return None
    try:
        raw = joblib.load(path)
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar {path}: {e}")
        return None
    # If it's a dict likely saved with multiple items; try to extract common keys
    if isinstance(raw, dict):
        for k in ("model", "pipeline", "estimator", "clf", "clf_cw", "bmi"):
            if k in raw:
                return raw[k]
        # fallback: return first value
        try:
            return list(raw.values())[0]
        except Exception:
            return raw
    return raw

def to_float(v, default):
    if v is None:
        return default
    if isinstance(v, str):
        v = v.replace(",", ".")
    try:
        return float(v)
    except Exception:
        return default

def prepare_bmi_df_for_model(model, bmi_val):
    if model is None:
        return None
    # if dict then extract inner model
    m = model
    if isinstance(m, dict):
        try:
            m = list(m.values())[0]
        except Exception:
            pass
    cols = getattr(m, "feature_names_in_", None)
    if cols is None:
        # fallback single BMI
        return pd.DataFrame([[bmi_val]], columns=["BMI"])
    cols = list(cols)
    # create row: put bmi_val into any column that has 'bmi' in the name, else 0
    row = {}
    for c in cols:
        row[c] = bmi_val if "bmi" in c.lower() else 0
    X = pd.DataFrame([row])
    if len(X.columns) == 1:
        X.columns = ["BMI"]
    return X

def adapt_for_model(model, X):
    if model is None:
        return None
    m = model
    if isinstance(m, dict):
        try:
            m = list(m.values())[0]
        except Exception:
            pass
    names = getattr(m, "feature_names_in_", None)
    if names is not None:
        names = list(names)
        # add missing columns with default 0 or sensible default
        for c in names:
            if c not in X.columns:
                X[c] = 0
        return X[names]
    return X

def extract_classes(mdl):
    if mdl is None:
        return None
    try:
        return getattr(mdl, "classes_", None)
    except Exception:
        return None

def save_csv_append(path: Path, df: pd.DataFrame):
    header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode='a', header=header, index=False, encoding='utf-8')

def append_feedback(row: dict, path: Path = FEEDBACK_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if header:
            writer.writeheader()
        writer.writerow(row)

def make_md_report(inputs: dict, pred_bmi, prob_bmi, pred_multi, prob_multi, final, reason):
    md = "# Relatório de Predição\n\n"
    md += f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    md += f"- IMC calculado: {inputs.get('BMI', 0):.2f}\n"
    md += f"- Predição (BMI-only): {pred_bmi} (prob: {prob_bmi})\n"
    md += f"- Predição (multi): {pred_multi} (prob: {prob_multi})\n"
    md += f"- Decisão final: {final}\n"
    md += f"- Motivo: {reason}\n\n"
    md += "## Inputs\n\n"
    for k, v in inputs.items():
        md += f"- {k}: {v}\n"
    return md.encode("utf-8")

# ---------------- UI / App ----------------
ensure_models_dir()
st.title("Previsor de Obesidade — IMC (rápido) + Modelo Completo (opcional)")

# Sidebar: detected models & admin toggle
st.sidebar.header("Modelos detectados / Config")
bmi_path, multi_path, model_files = find_models(MODELS_DIR)
st.sidebar.write(model_files if model_files else "(nenhum .joblib em models/)")
if bmi_path:
    st.sidebar.success(f"Modelo BMI detectado: {bmi_path.name}")
else:
    st.sidebar.warning("Modelo BMI não detectado.")
if multi_path:
    st.sidebar.success(f"Modelo completo detectado: {multi_path.name}")
else:
    st.sidebar.warning("Modelo completo não detectado.")

show_admin = st.sidebar.checkbox("Mostrar painel admin", value=False)

# Load models (cached)
bmi_model = safe_load_model(bmi_path) if bmi_path else None
multi_model = safe_load_model(multi_path) if multi_path else None

# sample defaults from dataset if available
sample = {}
if DATA_PATH.exists():
    try:
        sample_df = pd.read_csv(DATA_PATH)
        if not sample_df.empty:
            sample = sample_df.iloc[0].to_dict()
    except Exception:
        sample = {}

# map options (Portuguese display -> model expected)
map_sim_nao = {"Sim": "yes", "Não": "no", "Nao": "no", "Yes":"yes", "No":"no", "yes":"yes", "no":"no"}

# ---------- Section 1: IMC quick ----------
st.header("1) Resultado rápido — apenas Peso e Altura (IMC)")
col_w, col_h, col_b = st.columns([1.2, 1.0, 0.6])
with col_w:
    w_default = to_float(sample.get("Weight", 70), 70.0)
    weight = st.number_input("Peso (kg)", min_value=1.0, max_value=400.0, value=w_default, step=0.1, format="%.1f")
with col_h:
    h_default = to_float(sample.get("Height", 1.70), 1.70)
    height = st.number_input("Altura (m)", min_value=0.5, max_value=2.5, value=h_default, step=0.01, format="%.2f")
with col_b:
    bmi_val = weight / (height ** 2) if height > 0 else 0.0
    st.metric("IMC (BMI)", f"{bmi_val:.2f}")

st.markdown("---")
st.subheader("Resultado Rápido")
bmi_input_df = prepare_bmi_df_for_model(bmi_model, bmi_val)
pred_bmi, prob_bmi = None, None
if bmi_model is not None and bmi_input_df is not None:
    try:
        pred_bmi = bmi_model.predict(bmi_input_df)[0]
    except Exception as e:
        st.warning("Modelo BMI não aceitou a entrada — verifique pipeline. Erro: " + str(e))
    try:
        prob_arr = bmi_model.predict_proba(bmi_input_df)
        prob_bmi = float(np.max(prob_arr, axis=1)[0])
    except Exception:
        prob_bmi = None

if pred_bmi is not None:
    st.write(f"**Predição (BMI-only):** {pred_bmi}")
    if prob_bmi is not None:
        st.write(f"Confiança (max prob): {prob_bmi:.3f}")
else:
    st.info("Modelo BMI não disponível ou falhou. Verifique `models/`.")

# ---------- Section 2: Full form ----------
st.markdown("---")
st.header("2) Resultado completo (opcional) — responda mais perguntas para análise detalhada")

with st.expander("Mostrar formulário completo"):
    with st.form("form_completo"):
        st.markdown("**Informações básicas**")
        gender = st.selectbox("Gênero", options=["Female", "Male"], index=(0 if sample.get("Gender","Female")=="Female" else 1))
        age = st.number_input("Idade (anos)", min_value=1, max_value=120, value=int(sample.get("Age", 25) if sample else 25))

        st.markdown("**Hábitos alimentares**")
        family_history_simnao = st.selectbox("Histórico familiar de obesidade?", options=["Sim","Não"], index=(0 if sample.get("family_history","yes")=="yes" else 1))
        family_history = map_sim_nao.get(family_history_simnao, "no")

        favc_simnao = st.selectbox("Consumo frequente de alimentos calóricos?", options=["Sim","Não"], index=(0 if sample.get("FAVC","yes")=="yes" else 1))
        favc = map_sim_nao.get(favc_simnao, "no")

        fcvc = st.slider("Consumo de verduras (1=baixo → 5=alto)", 1, 5, value=int(sample.get("FCVC", 2) if sample else 2))
        ncp = st.slider("Número de refeições principais por dia", 1, 6, value=int(sample.get("NCP", 3) if sample else 3))
        caec = st.selectbox("Come entre refeições?", options=["no", "Sometimes", "Frequently", "Always"], index=0)

        st.markdown("**Estilo de vida**")
        smoke_simnao = st.selectbox("Fuma?", options=["Sim","Não"], index=0)
        smoke = map_sim_nao.get(smoke_simnao, "no")

        ch2o = st.slider("Consumo diário de água (1-5)", 1, 5, value=int(sample.get("CH2O", 2) if sample else 2))
        scc_simnao = st.selectbox("Monitora calorias regularmente?", options=["Sim","Não"], index=0)
        scc = map_sim_nao.get(scc_simnao, "no")

        faf = st.slider("Frequência de atividade física (1-5)", 1, 5, value=int(sample.get("FAF", 2) if sample else 2))
        tue = st.number_input("Horas por dia em telas/dispositivos (TUE)", min_value=0, max_value=24, value=int(sample.get("TUE", 2) if sample else 2))
        calc = st.selectbox("Consumo de álcool", options=["no", "Sometimes", "Frequently", "Always"], index=0)
        mtrans = st.selectbox("Transporte habitual", options=["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"], index=0)

        submitted = st.form_submit_button("Analisar (modelo completo)")

    if submitted:
        # assemble inputs
        inputs = {
            "Gender": gender,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "BMI": bmi_val,
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

        X_full = pd.DataFrame([inputs])
        X_for_model = adapt_for_model(multi_model, X_full.copy()) if multi_model is not None else None

        pred_multi, prob_multi = None, None
        if multi_model is not None:
            try:
                pred_multi = multi_model.predict(X_for_model)[0]
            except Exception as e:
                st.error("Erro ao predizer com o modelo completo: " + str(e))
                pred_multi = None
            try:
                prob_multi = float(np.max(multi_model.predict_proba(X_for_model), axis=1)[0])
            except Exception:
                prob_multi = None

        st.subheader("Resultado completo")
        if pred_multi is not None:
            st.write(f"**Predição (modelo completo):** {pred_multi}")
            if prob_multi is not None:
                st.write(f"Confiança (max prob): {prob_multi:.3f}")
        else:
            st.warning("Modelo completo não disponível ou falhou na predição.")

        # hybrid decision rule
        final, reason = None, ""
        if pred_bmi is None and pred_multi is None:
            st.error("Nenhum modelo disponível para decisão final.")
        elif pred_bmi is not None and pred_multi is None:
            final = pred_bmi; reason = "Somente IMC disponível (fallback)"
        elif pred_bmi is None and pred_multi is not None:
            final = pred_multi; reason = "Somente modelo completo disponível (fallback)"
        else:
            if pred_bmi == pred_multi:
                final = pred_bmi; reason = "Modelos concordam"
            else:
                # use probabilities if available
                if prob_bmi is None or prob_multi is None:
                    final = pred_bmi; reason = "Desempate: fallback IMC (probabilidades indisponíveis)"
                else:
                    delta = 0.10
                    if abs(prob_bmi - prob_multi) >= delta:
                        final = pred_bmi if prob_bmi > prob_multi else pred_multi
                        reason = f"Desempate por confiança (IMC {prob_bmi:.2f} vs Multi {prob_multi:.2f})"
                    else:
                        final = pred_bmi
                        reason = "Probabilidades semelhantes — fallback IMC"

        if final is not None:
            st.success(f"Decisão final: **{final}** — {reason}")

        # save discordance example if they disagree
        if pred_bmi is not None and pred_multi is not None and pred_bmi != pred_multi:
            out = X_full.copy()
            out['pred_bmi'] = pred_bmi
            out['prob_bmi'] = prob_bmi
            out['pred_multi'] = pred_multi
            out['prob_multi'] = prob_multi
            out['final'] = final
            out['reason'] = reason
            try:
                save_csv_append(DISCORDANCE_FILE, out)
            except Exception:
                # fallback append via pandas
                try:
                    DISCORDANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
                    out.to_csv(DISCORDANCE_FILE, mode='a', header=not DISCORDANCE_FILE.exists(), index=False)
                except Exception:
                    pass

        # feedback buttons
        st.markdown("### Seu feedback")
        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            if st.button("Este resultado está correto? ✅"):
                row = inputs.copy()
                row.update({
                    "pred_bmi": pred_bmi,
                    "prob_bmi": prob_bmi,
                    "pred_multi": pred_multi,
                    "prob_multi": prob_multi,
                    "final": final,
                    "reason": reason,
                    "feedback": "Sim",
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                try:
                    append_feedback(row)
                    st.success("Obrigado! Feedback salvo.")
                except Exception as e:
                    st.error("Falha ao salvar feedback: " + str(e))
        with fb_col2:
            if st.button("Este resultado NÃO está correto ❌"):
                row = inputs.copy()
                row.update({
                    "pred_bmi": pred_bmi,
                    "prob_bmi": prob_bmi,
                    "pred_multi": pred_multi,
                    "prob_multi": prob_multi,
                    "final": final,
                    "reason": reason,
                    "feedback": "Não",
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                try:
                    append_feedback(row)
                    st.success("Obrigado! Feedback salvo.")
                except Exception as e:
                    st.error("Falha ao salvar feedback: " + str(e))

        # allow download of a small report
        try:
            md = make_md_report(inputs, pred_bmi, prob_bmi, pred_multi, prob_multi, final, reason)
            st.download_button("Baixar relatório (MD)", data=md, file_name="predicao_report.md")
        except Exception:
            pass

# ---------- Admin panel ----------
st.markdown("---")
if show_admin:
    st.header("Painel admin / artifacts")
    colA, colB = st.columns([1,2])
    with colA:
        st.subheader("Arquivos em models/")
        files = sorted([p.name for p in MODELS_DIR.glob("*")]) if MODELS_DIR.exists() else []
        st.write(files)
        if DISCORDANCE_FILE.exists():
            st.markdown(f"- Discordâncias: `{DISCORDANCE_FILE}`")
            try:
                dfd = pd.read_csv(DISCORDANCE_FILE)
                st.dataframe(dfd.tail(50))
                csv_bytes = dfd.to_csv(index=False).encode("utf-8")
                st.download_button("Baixar discordâncias (CSV)", data=csv_bytes, file_name="discordance_examples.csv")
            except Exception as e:
                st.write("Erro ao ler discordance file: " + str(e))
        else:
            st.write("Nenhum exemplo de discordância salvo ainda.")
        if FEEDBACK_FILE.exists():
            st.markdown(f"- Feedbacks: `{FEEDBACK_FILE}`")
            try:
                dff = pd.read_csv(FEEDBACK_FILE)
                st.dataframe(dff.tail(50))
                csvb = dff.to_csv(index=False).encode("utf-8")
                st.download_button("Baixar feedbacks (CSV)", data=csvb, file_name="feedback.csv")
            except Exception as e:
                st.write("Erro ao ler feedback file: " + str(e))
        else:
            st.write("Nenhum feedback registrado ainda.")
    with colB:
        st.subheader("Model debug")
        st.write("BMI model:", bmi_path.name if bmi_path else "(não carregado)")
        st.write("Multi model:", multi_path.name if multi_path else "(não carregado)")
        st.write("Classes (BMI):", extract_classes(bmi_model))
        st.write("Classes (multi):", extract_classes(multi_model))
        st.write("Dataset local detectado em:", str(DATA_PATH))
        if DATA_PATH.exists():
            try:
                df_sample = pd.read_csv(DATA_PATH)
                st.write("Tamanho dataset (linhas):", len(df_sample))
                st.dataframe(df_sample.head(3))
            except Exception as e:
                st.write("Erro lendo dataset local: " + str(e))

# Footer guidance
st.markdown("---")
st.markdown(
    """
    **Observações**:
    - Se for ajustar rótulos (ex.: usar "Sim"/"Não" no UI), o app já faz o mapeamento interno para `yes/no`.
    - Para produção: considere calibrar probabilidades, registrar logs centralizados e auditar fairness por grupo.
    """
)

