# app_dual_model_pt_shap.py
"""
App Streamlit (PT) — IMC rápido + Modelo completo (opcional) + SHAP
- Tradução completa das perguntas para português.
- Mapeamento interno PT -> valores que o pipeline espera (ex: "Sim" -> "yes").
- Integração SHAP (opcional): gera explicações (summary + force plot) apenas quando solicitado.
- Salva feedback do usuário em models/feedback.csv e discordâncias em models/discordance_examples.csv.
- Usa dataset local como referência: /mnt/data/Obesity.csv (se existir) ou Obesity.csv.
Run: pip install shap matplotlib
     streamlit run app_dual_model_pt_shap.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os, time, csv
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="Previsor de Obesidade — IMC + Modelo Completo + SHAP", layout="wide")
st.title("Demo: IMC (rápido) + Modelo Completo (opcional) — com explicações SHAP")

# ---------- CONFIG ----------
MODELS_DIR = Path("models")
DATA_PATH = Path("/mnt/data/Obesity.csv") if Path("/mnt/data/Obesity.csv").exists() else Path("Obesity.csv")
DISCORDANCE_FILE = MODELS_DIR / "discordance_examples.csv"
FEEDBACK_FILE = MODELS_DIR / "feedback.csv"

# ---------- HELPERS ----------
def ensure_models_dir():
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

def find_models(models_dir):
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return None, None, []
    files = [f.name for f in models_dir.glob("*.joblib")]
    bmi_path = None
    multi_path = None
    for f in files:
        lf = f.lower()
        if "bmi" in lf and bmi_path is None:
            bmi_path = models_dir / f
        if any(x in lf for x in ("multi","no_bmi","class_weight","pipeline")) and multi_path is None and (bmi_path is None or f != bmi_path.name):
            multi_path = models_dir / f
    # fallback
    if bmi_path is None and files:
        for f in files:
            if "bmi" in f.lower():
                bmi_path = models_dir / f
                break
    return bmi_path, multi_path, files

@st.cache_resource
def load_model_safe(path):
    if not path:
        return None
    try:
        raw = joblib.load(path)
    except Exception as e:
        st.sidebar.error(f"Erro carregando {path}: {e}")
        return None
    # if dict try to guess key
    if isinstance(raw, dict):
        for k in ("model","pipeline","clf","estimator","clf_cw"):
            if k in raw:
                return raw[k]
        try:
            return list(raw.values())[0]
        except Exception:
            return raw
    return raw

def to_float(v, default=0.0):
    if pd.isna(v):
        return default
    if isinstance(v, str):
        v = v.replace(",", ".")
    try:
        return float(v)
    except Exception:
        return default

def prepare_bmi_df_for_model(model, bmi_val):
    """Return dataframe compatible with BMI-only model. If the model exposes feature_names_in_,
    insert BMI into the column that contains 'bmi', else create single-column DataFrame."""
    if model is None:
        return pd.DataFrame([[bmi_val]], columns=["BMI"])
    m = model
    if isinstance(m, dict):
        # fallback
        try:
            m = list(m.values())[0]
        except Exception:
            m = m
    cols = getattr(m, "feature_names_in_", None)
    if cols is None:
        return pd.DataFrame([[bmi_val]], columns=["BMI"])
    cols = list(cols)
    row = {}
    assigned = False
    for c in cols:
        if "bmi" in c.lower() and not assigned:
            row[c] = bmi_val
            assigned = True
        else:
            row[c] = 0
    return pd.DataFrame([row])

def adapt_for_model(model, X_df):
    """Ensure X_df contains the model's feature_names_in_ columns in the right order (fill missing with 0)."""
    if model is None:
        return None
    m = model
    if isinstance(m, dict):
        try:
            m = list(m.values())[0]
        except Exception:
            m = m
    names = getattr(m, "feature_names_in_", None)
    if names is None:
        return X_df
    names = list(names)
    for c in names:
        if c not in X_df.columns:
            X_df[c] = 0
    return X_df[names]

def save_csv_append(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
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

def get_tree_estimator_from_pipeline(model):
    """Try to extract a tree-based estimator (RandomForest/DecisionTree) from a pipeline or estimator."""
    if model is None:
        return None
    # if pipeline-like
    if hasattr(model, "named_steps"):
        for name, step in model.named_steps.items():
            if hasattr(step, "feature_importances_") or "RandomForest" in type(step).__name__:
                return step
        # fallback: last step
        try:
            last = list(model.named_steps.values())[-1]
            return last
        except Exception:
            return None
    # if ensemble or tree itself
    if hasattr(model, "feature_importances_") or "RandomForest" in type(model).__name__:
        return model
    # else none
    return None

# ---------- Setup / Load models ----------
ensure_models_dir()
bmi_path, multi_path, model_files = find_models(MODELS_DIR)
st.sidebar.header("Modelos detectados")
st.sidebar.write(model_files if model_files else "(nenhum .joblib em models/)")
if bmi_path:
    st.sidebar.success(f"Modelo IMC: {bmi_path.name}")
else:
    st.sidebar.warning("Modelo IMC não detectado.")
if multi_path:
    st.sidebar.success(f"Modelo completo: {multi_path.name}")
else:
    st.sidebar.warning("Modelo completo não detectado.")

bmi_model = load_model_safe(bmi_path) if bmi_path else None
multi_model = load_model_safe(multi_path) if multi_path else None

# ---------- sample dataset to populate defaults ----------
sample = {}
if DATA_PATH.exists():
    try:
        df_full = pd.read_csv(DATA_PATH)
        if not df_full.empty:
            sample = df_full.iloc[0].to_dict()
    except Exception:
        sample = {}

# ---------- mapping PT -> model values ----------
map_sim_nao = {"Sim":"yes", "Não":"no", "Nao":"no", "Sim":"yes", "Não":"no"}
# For categorical text values used in dataset we map PT options to original expected tokens
map_caec = {
    "Raramente": "no",
    "Às vezes": "Sometimes",
    "Freqüentemente": "Frequently",
    "Sempre": "Always",
    # fallback english
    "no":"no","Sometimes":"Sometimes","Frequently":"Frequently","Always":"Always"
}
map_calc = {"Não":"no","Raramente":"no","Às vezes":"Sometimes","Freqüentemente":"Frequently","Sempre":"Always",
            "no":"no","Sometimes":"Sometimes","Frequently":"Frequently","Always":"Always"}
map_mtrans = {
    "Transporte público": "Public_Transportation",
    "A pé": "Walking",
    "Automóvel": "Automobile",
    "Moto": "Motorbike",
    "Bicicleta": "Bike"
}

# ---------- UI: quick IMC ----------
st.header("1) Resultado rápido — Peso e Altura (IMC)")
col1, col2, col3 = st.columns([1.2,1.0,0.6])
with col1:
    weight = st.number_input("Peso (kg)", min_value=1.0, max_value=400.0,
                             value=float(sample.get("Weight", 70)), step=0.1, format="%.1f")
with col2:
    height = st.number_input("Altura (m)", min_value=0.5, max_value=2.5,
                             value=float(sample.get("Height", 1.70)), step=0.01, format="%.2f")
with col3:
    bmi_val = weight / (height ** 2) if height > 0 else 0.0
    st.metric("IMC (BMI)", f"{bmi_val:.2f}")

# Try to predict with bmi_model
bmi_input_df = prepare_bmi_df_for_model(bmi_model, bmi_val)
pred_bmi = None
prob_bmi = None
if bmi_model is not None:
    try:
        pred_bmi = bmi_model.predict(bmi_input_df)[0]
    except Exception as e:
        st.warning(f"Falha ao prever com modelo BMI: {e}")
    try:
        prob_bmi = float(np.max(bmi_model.predict_proba(bmi_input_df), axis=1)[0])
    except Exception:
        prob_bmi = None

if pred_bmi is not None:
    st.write(f"**Predição (IMC-only):** {pred_bmi}")
    if prob_bmi is not None:
        st.write(f"Confiança (probabilidade máxima): {prob_bmi:.3f}")
else:
    st.info("Modelo IMC não carregado ou falhou. Verifique pasta `models/`.")

st.markdown("---")

# ---------- UI: formulário completo (PT) ----------
st.header("2) Resultado completo (opcional) — responda mais perguntas")
with st.expander("Mostrar formulário completo"):
    with st.form("form_comp"):
        st.subheader("Informações básicas")
        gender = st.selectbox("Gênero", options=["Feminino","Masculino"], index=(0 if str(sample.get("Gender","Female")).lower().startswith("f") else 1))
        age = st.number_input("Idade (anos)", min_value=1, max_value=120, value=int(sample.get("Age", 25)))

        st.subheader("Hábitos alimentares")
        family_history_pt = st.selectbox("Histórico familiar de obesidade?", options=["Sim","Não"], index=(0 if sample.get("family_history","yes")=="yes" else 1))
        favc_pt = st.selectbox("Consumo frequente de alimentos calóricos?", options=["Sim","Não"], index=(0 if sample.get("FAVC","yes")=="yes" else 1))
        fcvc = st.slider("Consumo de verduras (1=baixo → 5=alto)", 1, 5, value=int(sample.get("FCVC", 2)))
        ncp = st.slider("Número de refeições principais por dia", 1, 6, value=int(sample.get("NCP", 3)))
        caec_pt = st.selectbox("Come entre refeições?", options=["Raramente","Às vezes","Freqüentemente","Sempre"], index=1)

        st.subheader("Estilo de vida")
        smoke_pt = st.selectbox("Fuma?", options=["Sim","Não"], index=1)
        ch2o = st.slider("Consumo diário de água (1-5)", 1, 5, value=int(sample.get("CH2O", 2)))
        scc_pt = st.selectbox("Monitora calorias regularmente?", options=["Sim","Não"], index=1)
        faf = st.slider("Frequência de atividade física (1-5)", 1, 5, value=int(sample.get("FAF", 2)))
        tue = st.number_input("Horas por dia em telas/dispositivos (TUE)", min_value=0, max_value=24, value=int(sample.get("TUE", 2)))
        calc_pt = st.selectbox("Consumo de álcool", options=["Não","Raramente","Às vezes","Freqüentemente","Sempre"], index=0)
        mtrans_pt = st.selectbox("Transporte habitual", options=["Transporte público","A pé","Automóvel","Moto","Bicicleta"], index=0)

        submitted = st.form_submit_button("Analisar (modelo completo)")

    if submitted:
        # map PT-valued fields into model-expected tokens
        family_history = map_sim_nao.get(family_history_pt, "no")
        favc = map_sim_nao.get(favc_pt, "no")
        caec = map_caec.get(caec_pt, "Sometimes")
        smoke = map_sim_nao.get(smoke_pt, "no")
        scc = map_sim_nao.get(scc_pt, "no")
        calc = map_calc.get(calc_pt, "no")
        mtrans = map_mtrans.get(mtrans_pt, "Public_Transportation")
        # gender mapping to original token (dataset uses Male/Female)
        gender_token = "Female" if gender == "Feminino" else "Male"

        inputs = {
            "Gender": gender_token,
            "Age": int(age),
            "Height": float(height),
            "Weight": float(weight),
            "BMI": float(bmi_val),
            "family_history": family_history,
            "FAVC": favc,
            "FCVC": int(fcvc),
            "NCP": int(ncp),
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": int(ch2o),
            "SCC": scc,
            "FAF": int(faf),
            "TUE": int(tue),
            "CALC": calc,
            "MTRANS": mtrans
        }

        # prepare X for multi_model
        X_full = pd.DataFrame([inputs])
        X_for_model = adapt_for_model(multi_model, X_full.copy()) if multi_model is not None else None

        pred_multi = None
        prob_multi = None
        if multi_model is not None and X_for_model is not None:
            try:
                pred_multi = multi_model.predict(X_for_model)[0]
            except Exception as e:
                st.error("Erro ao prever com o modelo completo: " + str(e))
            try:
                prob_multi = float(np.max(multi_model.predict_proba(X_for_model), axis=1)[0])
            except Exception:
                prob_multi = None

        st.subheader("Resultado completo")
        if pred_multi is not None:
            st.write(f"**Predição (modelo completo):** {pred_multi}")
            if prob_multi is not None:
                st.write(f"Confiança (probabilidade máxima): {prob_multi:.3f}")
        else:
            st.warning("Modelo completo não disponível ou apresentou erro.")

        # hybrid decision (same logic prior)
        final, reason = None, ""
        if pred_bmi is None and pred_multi is None:
            st.error("Nenhum modelo disponível para decisão final.")
        elif pred_bmi is not None and pred_multi is None:
            final = pred_bmi; reason = "Modelo completo indisponível — fallback IMC"
        elif pred_bmi is None and pred_multi is not None:
            final = pred_multi; reason = "Modelo IMC indisponível — fallback modelo completo"
        else:
            if pred_bmi == pred_multi:
                final = pred_bmi; reason = "Modelos concordam"
            else:
                # use probabilities if available
                if prob_bmi is None or prob_multi is None:
                    final = pred_bmi; reason = "Probabilidades indisponíveis — fallback IMC"
                else:
                    delta = 0.10
                    if abs(prob_bmi - prob_multi) >= delta:
                        final = pred_bmi if prob_bmi > prob_multi else pred_multi
                        reason = f"Desempate por confiança (IMC {prob_bmi:.2f} vs Multi {prob_multi:.2f})"
                    else:
                        final = pred_bmi; reason = "Probabilidades similares — fallback IMC"

        if final is not None:
            st.success(f"Decisão final: **{final}** — {reason}")

        # save discordance sample
        if pred_bmi is not None and pred_multi is not None and pred_bmi != pred_multi:
            try:
                out = X_full.copy()
                out['pred_bmi'] = pred_bmi
                out['prob_bmi'] = prob_bmi
                out['pred_multi'] = pred_multi
                out['prob_multi'] = prob_multi
                out['final'] = final
                out['reason'] = reason
                save_csv_append(DISCORDANCE_FILE, out)
            except Exception:
                pass

        # feedback buttons
        st.markdown("### Seu feedback (ajuda a melhorar o modelo)")
        fbcol1, fbcol2 = st.columns(2)
        with fbcol1:
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
                append_feedback(row)
                st.success("Obrigado! Feedback salvo em models/feedback.csv")
        with fbcol2:
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
                append_feedback(row)
                st.success("Obrigado! Feedback salvo em models/feedback.csv")

        # report download
        try:
            md = make_md_report(inputs, pred_bmi, prob_bmi, pred_multi, prob_multi, final, reason)
            st.download_button("Baixar relatório (MD)", data=md, file_name="predicao_report.md")
        except Exception:
            pass

        # ---------- SHAP area ----------
        st.markdown("---")
        st.subheader("Explicação local (SHAP) — opcional")
        st.markdown(
            "Calcule explicações SHAP para entender quais features mais contribuíram para a predição. "
            "Aviso: o cálculo pode ser pesado dependendo do modelo e do background sample."
        )
        use_shap = st.checkbox("Gerar explicações SHAP para esta predição (pode demorar)")
        if use_shap:
            try:
                import shap
                model_for_shap = get_tree_estimator_from_pipeline(multi_model)
                if model_for_shap is None:
                    st.warning("Não foi possível extrair um estimador de árvore do pipeline para SHAP. SHAP TreeExplainer funciona melhor com RandomForest/DecisionTree.")
                else:
                    # background/sample data for SHAP: take small sample from dataset or create artificial zeros
                    if 'df_full' in globals() and isinstance(df_full, pd.DataFrame) and not df_full.empty:
                        # adapt background to model
                        try:
                            bg = adapt_for_model(multi_model, df_full.sample(min(100, len(df_full)), random_state=42).copy())
                        except Exception:
                            bg = None
                    else:
                        bg = None
                    # prepare single-row X_for_shap compatible
                    if X_for_model is None:
                        st.error("Entrada inválida para o modelo — não é possível gerar SHAP.")
                    else:
                        with st.spinner("Calculando SHAP explainer..."):
                            explainer = shap.TreeExplainer(model_for_shap, data=bg if bg is not None else None)
                            # shap values (for multiclass returns list)
                            shap_values = explainer.shap_values(adapt_for_model(multi_model, X_for_model.copy()))
                        # choose class index = predicted class
                        class_idx = None
                        try:
                            class_labels = getattr(model_for_shap, "classes_", getattr(multi_model, "classes_", None))
                            if class_labels is not None and pred_multi is not None:
                                if isinstance(class_labels, np.ndarray):
                                    labels = list(class_labels)
                                else:
                                    labels = list(class_labels)
                                try:
                                    class_idx = labels.index(pred_multi)
                                except Exception:
                                    class_idx = None
                        except Exception:
                            class_idx = None
                        st.write("Visualizações SHAP:")
                        # summary (for class or overall)
                        try:
                            plt.figure(figsize=(8,4))
                            if isinstance(shap_values, list):
                                if class_idx is None:
                                    # show summary for first class
                                    shap.summary_plot(shap_values[0], adapt_for_model(multi_model, X_for_model.copy()), show=False)
                                else:
                                    shap.summary_plot(shap_values[class_idx], adapt_for_model(multi_model, X_for_model.copy()), show=False)
                            else:
                                shap.summary_plot(shap_values, adapt_for_model(multi_model, X_for_model.copy()), show=False)
                            st.pyplot(plt.gcf())
                            plt.clf()
                        except Exception as e:
                            st.write("Erro ao gerar summary_plot SHAP:", e)
                        # force_plot (local)
                        try:
                            st.markdown("#### Força local (force plot) — contributions por feature")
                            if isinstance(shap_values, list) and class_idx is not None:
                                sv = shap_values[class_idx]
                            else:
                                sv = shap_values
                            # force_plot returns JS html; streamlit can render via components
                            force_html = shap.plots.force(explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[class_idx],
                                                          sv[0], adapt_for_model(multi_model, X_for_model.copy()).iloc[0], matplotlib=False, show=False)
                            # shap's force returns a JS widget; shap can write it to HTML
                            try:
                                from streamlit.components.v1 import html
                                html(shap.plots._js.html())  # best-effort; some shap versions differ
                                st.write("Se o widget acima não apareceu, verifique a versão do SHAP / browser.")
                            except Exception:
                                st.write("Force plot interativo pode não ser renderizado corretamente nesta configuração.")
                        except Exception as e:
                            st.write("Erro ao gerar force_plot:", e)
            except Exception as e:
                st.error("SHAP não está disponível. Instale com `pip install shap matplotlib` e tente novamente. Erro: " + str(e))

# ---------- Admin / artifacts ----------
st.markdown("---")
st.header("Admin / Artefatos")
colA, colB = st.columns([1,2])
with colA:
    st.subheader("Conteúdo de models/")
    if MODELS_DIR.exists():
        files = sorted([p.name for p in MODELS_DIR.glob("*")])
        st.write(files)
    else:
        st.write("(pasta models/ vazia)")
    if DISCORDANCE_FILE.exists():
        try:
            d = pd.read_csv(DISCORDANCE_FILE)
            st.write("Exemplos de discordância (ultimas 20):")
            st.dataframe(d.tail(20))
            st.download_button("Baixar discordâncias (CSV)", data=d.to_csv(index=False).encode("utf-8"), file_name="discordance_examples.csv")
        except Exception as e:
            st.write("Erro lendo discordance file:", e)
    else:
        st.write("Nenhuma discordância ainda.")
    if FEEDBACK_FILE.exists():
        try:
            ff = pd.read_csv(FEEDBACK_FILE)
            st.write("Feedbacks (ultimos 20):")
            st.dataframe(ff.tail(20))
            st.download_button("Baixar feedbacks (CSV)", data=ff.to_csv(index=False).encode("utf-8"), file_name="feedback.csv")
        except Exception as e:
            st.write("Erro lendo feedback file:", e)
    else:
        st.write("Nenhum feedback registrado ainda.")
with colB:
    st.subheader("Debug modelos")
    st.write("Modelo IMC:", bmi_path.name if bmi_path else "(não carregado)")
    st.write("Modelo completo:", multi_path.name if multi_path else "(não carregado)")
    try:
        st.write("Classes (IMC):", getattr(bmi_model, "classes_", None))
    except Exception:
        st.write("Classes (IMC): -")
    try:
        st.write("Classes (Multi):", getattr(multi_model, "classes_", None))
    except Exception:
        st.write("Classes (Multi): -")
    st.write("Dataset local (referência):", str(DATA_PATH))
    if DATA_PATH.exists():
        try:
            df_info = pd.read_csv(DATA_PATH)
            st.write("Tamanho dataset:", len(df_info))
            st.dataframe(df_info.head(3))
            st.download_button("Baixar dataset usado (referência)", data=df_info.to_csv(index=False).encode("utf-8"), file_name="Obesity_reference.csv")
        except Exception as e:
            st.write("Erro lendo dataset:", e)

st.markdown("---")
st.info("Se preferir que eu ajuste os rótulos textuais (ex.: 'Freqüentemente' -> 'Frequent') para exatamente os tokens do seu pipeline, me diga quais tokens seu pipeline espera. Também posso ajustar o tamanho do background sample do SHAP para ficar mais leve.")

# EOF
