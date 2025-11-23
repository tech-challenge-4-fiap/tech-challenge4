# compare_models.py (versão final — inspeciona pre.transformers_ e monta inputs com tipos corretos)
import joblib, os, sys
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
try:
    from statsmodels.stats.contingency_tables import mcnemar
    HAS_MCNEMAR = True
except Exception:
    HAS_MCNEMAR = False
    print("Aviso: statsmodels não disponível. McNemar não será executado.")

DATA_PATH = "Obesity.csv"
MODELS_DIR = "models"

if not os.path.exists(MODELS_DIR):
    sys.exit(f"Diretório {MODELS_DIR} não encontrado. Rode o treino primeiro.")

joblib_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
print("Arquivos .joblib encontrados em models/:", joblib_files)
if not joblib_files:
    sys.exit("Nenhum .joblib encontrado no diretório models/")

# heurística para escolher BMI-only e Multi
bmi_candidate = None
multi_candidate = None
for f in joblib_files:
    lf = f.lower()
    if "bmi" in lf or "bmi_pipeline" in lf:
        bmi_candidate = os.path.join(MODELS_DIR, f)
    elif "class_weight" in lf or "smote" in lf or ("pipeline" in lf and "bmi" not in lf):
        if multi_candidate is None:
            multi_candidate = os.path.join(MODELS_DIR, f)

if bmi_candidate is None:
    sorted_files = sorted(joblib_files, key=lambda x: os.path.getsize(os.path.join(MODELS_DIR,x)))
    bmi_candidate = os.path.join(MODELS_DIR, sorted_files[0])
    if len(sorted_files) > 1:
        multi_candidate = os.path.join(MODELS_DIR, sorted_files[-1])
else:
    if multi_candidate is None:
        others = [f for f in joblib_files if os.path.join(MODELS_DIR,f) != bmi_candidate]
        if others:
            multi_candidate = os.path.join(MODELS_DIR, others[0])

print("BMI candidate:", bmi_candidate)
print("Multi candidate:", multi_candidate)

# load data
if not os.path.exists(DATA_PATH):
    if os.path.exists("/mnt/data/Obesity.csv"):
        DATA_PATH = "/mnt/data/Obesity.csv"
    else:
        sys.exit(f"Arquivo de dados {DATA_PATH} não encontrado. Coloque Obesity.csv na raiz do projeto.")
print("Usando DATA_PATH:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
target_col = 'Obesity' if 'Obesity' in df.columns else df.columns[-1]
if 'BMI' not in df.columns and 'Weight' in df.columns and 'Height' in df.columns:
    df['BMI'] = df['Weight'] / (df['Height']**2)
y = df[target_col].astype(str)
X = df.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def unwrap_obj(obj):
    if isinstance(obj, dict):
        return obj
    return obj

def load_model_safe(path):
    print("Carregando:", path)
    raw = joblib.load(path)
    print(" -> tipo raw:", type(raw))
    return unwrap_obj(raw)

m_bmi_raw = load_model_safe(bmi_candidate)
m_multi_raw = load_model_safe(multi_candidate) if multi_candidate else None

# util: default value by column with correct type handling
def default_for_column(colname, X_ref):
    if colname in X_ref.columns:
        ser = X_ref[colname]
        if pd.api.types.is_numeric_dtype(ser):
            return float(ser.mean())
        else:
            modes = ser.mode(dropna=True)
            if not modes.empty:
                return modes.iloc[0]
            nonnull = ser.dropna()
            if not nonnull.empty:
                return nonnull.iloc[0]
            return ""
    # heuristics for unknown column names
    if "_" in colname or "-" in colname or colname.lower().endswith(("0","1")):
        return 0
    if 'gender' in colname.lower() or 'sex' in colname.lower() or 'genero' in colname.lower():
        for c in X_ref.columns:
            if c.lower() in ('gender','sex','genero'):
                modes = X_ref[c].mode(dropna=True)
                return modes.iloc[0] if not modes.empty else ""
    return 0

# --- substitua por isto ---
def build_row_for_pre(pre, bmi_val, X_ref):
    """
    Constrói uma linha de entrada (DataFrame 1xN) para passar ao pre.transform()
    usando um prototype row existente em X_ref para preservar dtypes corretos.
    """
    # 1) pegar uma linha "prototype" existente do dataset (mantém dtypes)
    prototype = X_ref.iloc[[0]].copy()  # DataFrame 1xN

    # 2) ajustar numericamente: para colunas numéricas podemos colocar a média (opcional)
    for c in prototype.columns:
        if pd.api.types.is_numeric_dtype(X_ref[c]):
            prototype.at[prototype.index[0], c] = float(X_ref[c].mean())
        else:
            # para categóricas, deixamos o valor do prototype (modo) — já tem tipo correto
            pass

    # 3) sobrescreve uma coluna BMI se existir, senão tenta injetar em coluna com nome parecido
    if 'BMI' in prototype.columns:
        prototype.at[prototype.index[0], 'BMI'] = float(bmi_val)
    else:
        bmi_cols = [col for col in prototype.columns if 'bmi' in col.lower() or 'imc' in col.lower()]
        if bmi_cols:
            prototype.at[prototype.index[0], bmi_cols[0]] = float(bmi_val)
        else:
            # se não existir coluna BMI, tenta adicionar 'BMI' (alguns preprocessors calculam internamente -> ok)
            prototype['BMI'] = float(bmi_val)

    # 4) coerce numéricos coerentemente (garantir floats onde o X_ref é numérico)
    for c in prototype.columns:
        if c in X_ref.columns and pd.api.types.is_numeric_dtype(X_ref[c]):
            prototype[c] = pd.to_numeric(prototype[c], errors='coerce').fillna(0)

    return prototype
# --- fim da substituição ---

def make_predict_fn(raw_obj):
    if not isinstance(raw_obj, dict):
        mdl = raw_obj
        def predict_fn(bmi_vals, X_ref):
            X_arr = np.array(bmi_vals).reshape(-1,1)
            try:
                return mdl.predict(X_arr)
            except Exception:
                pass
            try:
                Xdf = pd.DataFrame(np.array(bmi_vals).reshape(-1,1), columns=['BMI'])
                return mdl.predict(Xdf)
            except Exception:
                pass
            if hasattr(mdl, 'feature_names_in_'):
                names = list(mdl.feature_names_in_)
                defaults = {c: default_for_column(c, X_ref) for c in names}
                Xdf = pd.DataFrame([defaults], columns=names)
                for c in Xdf.columns:
                    if c in X_ref.columns and pd.api.types.is_numeric_dtype(X_ref[c]):
                        Xdf[c] = pd.to_numeric(Xdf[c], errors='coerce').fillna(0)
                return mdl.predict(Xdf)
            raise RuntimeError("Modelo não aceita entrada simples e não expõe feature_names_in_")
        return predict_fn

    # dict case: detect preprocessor + classifier keys
    keys = {k.lower(): k for k in raw_obj.keys()}
    pre_key = None
    clf_key = None
    for k in keys:
        if 'pre' in k or 'preprocessor' in k:
            pre_key = keys[k]
        if 'clf' in k or 'classifier' in k or 'model' in k:
            if 'clf_cw' in k or 'clf_sm' in k:
                clf_key = keys[k]
            elif clf_key is None:
                clf_key = keys[k]

    # case when pre+clf exist as separate objects in dict
    if pre_key and clf_key:
        pre = raw_obj[pre_key]
        clf = raw_obj[clf_key]
        print(f"Dict detectado com preprocessor key='{pre_key}' e classifier key='{clf_key}'")
        def predict_fn(bmi_vals, X_ref):
            # build Xrow using pre.transformers_ guidance
            # use first value of bmi_vals for single-row predict
            bmi_val = float(bmi_vals[0])
            Xrow = build_row_for_pre(pre, bmi_val, X_ref)
            # try transform
            try:
                X_trans = pre.transform(Xrow)
            except Exception:
                # coerce numeric columns again and retry
                Xrow2 = Xrow.copy()
                for c in Xrow2.columns:
                    if c in X_ref.columns and pd.api.types.is_numeric_dtype(X_ref[c]):
                        Xrow2[c] = pd.to_numeric(Xrow2[c], errors='coerce').fillna(0)
                X_trans = pre.transform(Xrow2)
            return clf.predict(X_trans)
        return predict_fn

    # fallback: try any dict value with predict
    for k,v in raw_obj.items():
        if hasattr(v, 'predict'):
            mdl = v
            print(f"Fallback: using dict value '{k}' as model")
            def predict_fn(bmi_vals, X_ref):
                X_arr = np.array(bmi_vals).reshape(-1,1)
                try:
                    return mdl.predict(X_arr)
                except Exception:
                    try:
                        Xdf = pd.DataFrame(np.array(bmi_vals).reshape(-1,1), columns=['BMI'])
                        return mdl.predict(Xdf)
                    except Exception:
                        raise RuntimeError("Fallback model did not accept BMI input")
            return predict_fn

    raise RuntimeError("Dict no modelo sem pre+clf detectáveis e sem valor com .predict()")

predict_bmi_fn = make_predict_fn(m_bmi_raw)
predict_multi_fn = make_predict_fn(m_multi_raw) if m_multi_raw is not None else None

# executar predições
bmi_vals = X_test['BMI'].values
print("Gerando predições BMI-model...")
pred_bmi = predict_bmi_fn(bmi_vals, X)
print("OK.")
if predict_multi_fn is not None:
    print("Gerando predições multi-model...")
    pred_multi = predict_multi_fn(bmi_vals, X)
    print("OK.")
else:
    pred_multi = None
    print("Nenhum modelo multi disponível.")

def summarize(y_true, y_pred, label):
    print("\n=== {} ===\n".format(label))
    print(classification_report(y_true, y_pred))
    print("Confusion:\n", confusion_matrix(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Macro-F1:", f1_score(y_true, y_pred, average='macro'))

summarize(y_test, pred_bmi, "BMI-only model")
if pred_multi is not None:
    summarize(y_test, pred_multi, "Multi-feature model")

if pred_multi is not None and HAS_MCNEMAR:
    y_true = y_test.values
    cw = (pred_bmi == y_true)
    sm = (pred_multi == y_true)
    b = int(((cw==True) & (sm==False)).sum())
    c = int(((cw==False) & (sm==True)).sum())
    table = [[int(((cw==True)&(sm==True)).sum()), b],[c, int(((cw==False)&(sm==False)).sum())]]
    print("\nMcNemar 2x2 table:", table)
    res = mcnemar(table, exact=False, correction=True)
    print("McNemar stat:", res.statistic, "pvalue:", res.pvalue)
elif pred_multi is not None:
    print("\nMcNemar não executado (statsmodels ausente).")
else:
    print("\nComparação não possível (modelo multi ausente).")

# per-group by Gender if available
group_col = None
for col in X_test.columns:
    if col.lower() in ('gender','sex','genero'):
        group_col = col
        break
if group_col and pred_multi is not None:
    print("\nPer-group metrics by", group_col)
    for g in X_test[group_col].unique():
        idx = X_test[group_col]==g
        if idx.sum() < 5: continue
        print("Group:", g)
        print(" BMI-only:", classification_report(y_test[idx], pd.Series(pred_bmi)[idx], zero_division=0))
        print(" Multi :", classification_report(y_test[idx], pd.Series(pred_multi)[idx], zero_division=0))
