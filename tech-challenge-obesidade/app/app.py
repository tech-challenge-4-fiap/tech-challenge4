import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# python -m streamlit run meu_app.py
# python -m streamlit run Predict_obesidade.py

# Ignorar warnings para uma saída mais limpa
warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO INICIAL E CSS (DESIGN) ---

# 1. Configurar a página (Wide Layout e Título)
st.set_page_config(
    layout="wide",
    page_title="Previsão de Obesidade (Foco ML)",
    page_icon="🤖"
)

# 2. Injetar CSS para Design
st.markdown("""
<style>
/* Fundo principal em cinza suave */
.stApp {
    background-color: #f0f2f6; 
}

/* Título principal em destaque */
h1 {
    color: #1f77b4; 
    text-align: center;
}

/* Títulos de seção */
h2 {
    border-bottom: 2px solid #e1e4e8;
    padding-bottom: 5px;
    margin-top: 25px;
    color: #333333;
}

/* Estilo para containers/cards */
.stContainer {
    background-color: white !important;
    border-radius: 10px;
    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.1);
    padding: 20px;
    margin-bottom: 20px;
}

/* Estilo para as métricas */
[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e6e6e6;
    box-shadow: 0 2px 4px 0 rgba(0,0,0,0.05);
}

/* Estilização da Sidebar */
.css-1d391kg { /* Target for sidebar content */
    background-color: #ffffff; /* Fundo branco para a sidebar */
}

</style>
""", unsafe_allow_html=True)


# Este título fica na página principal
st.markdown("<h1 style='text-align: center;'>Análise e Previsão de Nível de Obesidade (Modelo ML)</h1>", unsafe_allow_html=True)
st.write("---")

# ===================================================================
# 1. CARREGAR E TREINAR MODELO (ETAPA CRÍTICA)
# ===================================================================

from pathlib import Path

@st.cache_resource
def load_and_train_model():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR.parent / "data" / "Obesity.csv"

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {DATA_PATH}. Coloque o Obesity.csv em tech-challenge-obesidade/data/")
        st.stop()

        
    # Salvar as colunas originais para o modelo antes de renomear e mapear
    df['MTRANS_Original'] = df['MTRANS']
    
    # 2. Renomear Colunas (para análise/display)
    df.rename(columns={
        'Gender': 'Genero', 'Age': 'Idade', 'Height': 'Altura', 'Weight': 'Peso',
        'family_history': 'Histórico', 'SMOKE': 'Fuma?', 'FAF': 'Atividade_Fisica',
        'CALC': 'Alcool?', 'FAVC': 'Alimentos_Caloricos', 'FCVC': 'Vegetais?',
        'NCP': 'Refeicoes_principais', 'CAEC': 'Alimentos_entre_refeicoes?',
        'CH2O': 'Qtde_Agua', 'SCC': 'Monitora_calorias?', 'TUE': 'Tempo_tecnologia',
        'MTRANS': 'Transporte', 'Obesity': 'Nivel_Obesidade'
    }, inplace=True)

    # 3. Mapear Binários (para o Modelo: 1/0)
    rename_binary = {'yes': 1, 'no': 0}
    rename_binary_genero = {'Male': 1, 'Female': 0}
    df['Genero_Model'] = df['Genero'].map(rename_binary_genero) 
    df['Histórico_Model'] = df['Histórico'].map(rename_binary)
    df['Alimentos_Caloricos_Model'] = df['Alimentos_Caloricos'].map(rename_binary)
    df['Fuma?_Model'] = df['Fuma?'].map(rename_binary)
    df['Monitora_calorias?_Model'] = df['Monitora_calorias?'].map(rename_binary)

    # 4. Arredondamentos e Tipos
    df['Idade'] = df['Idade'].round(0).astype(int)
    df['Altura'] = df['Altura'].round(2)
    df['Peso'] = df['Peso'].round(1)
    df['Vegetais?'] = df['Vegetais?'].round(0).astype(int) 
    df['Refeicoes_principais'] = df['Refeicoes_principais'].round(0).astype(int)
    df['Qtde_Agua'] = df['Qtde_Agua'].round(0).astype(int)
    df['Atividade_Fisica'] = df['Atividade_Fisica'].round(0).astype(int)
    df['Tempo_tecnologia'] = df['Tempo_tecnologia'].round(0).astype(int)

    # 5. Feature Engineering (Criação do IMC)
    df['IMC'] = df['Peso'] / (df['Altura'] * df['Altura'])

    # 6. Mapear Target e Categoria para PT-BR
    rename_target = {
        'Normal_Weight': 'Peso Normal', 'Overweight_Level_I': 'Sobrepeso I',
        'Overweight_Level_II': 'Sobrepeso II', 'Obesity_Type_I': 'Obesidade I',
        'Insufficient_Weight': 'Peso Insuficiente', 'Obesity_Type_II': 'Obesidade II',
        'Obesity_Type_III': 'Obesidade III'
    }
    df['Nivel_Obesidade'] = df['Nivel_Obesidade'].map(rename_target)
    
    # Mapeamento para PT-BR (apenas para display na análise)
    rename_colunas_anl = {
        'Sometimes':'As vezes', 'Frequently':'Frequentemente', 'Always':'Sempre', 'no':'Não', 'yes': 'Sim'
    }
    # Aplicar mapeamento PT-BR
    df['Alimentos_entre_refeicoes?'] = df['Alimentos_entre_refeicoes?'].map(rename_colunas_anl)
    df['Alcool?'] = df['Alcool?'].map(rename_colunas_anl)
    
    # --- PRÉ-PROCESSAMENTO PARA O MODELO (Encoding Ordinal e Dummies) ---

    # 7. ENCODING ORDINAL (Usando as colunas em PT-BR para mapear para números)
    map_ordinal_model_pt = {'Não': 0, 'As vezes': 1, 'Frequentemente': 2, 'Sempre': 3}
    
    df['Alcool?_Model'] = df['Alcool?'].apply(lambda x: map_ordinal_model_pt.get(x, 0)) 
    df['Alimentos_entre_refeicoes?_Model'] = df['Alimentos_entre_refeicoes?'].apply(lambda x: map_ordinal_model_pt.get(x, 0))

    # 7b. Encoding Nominal (One-Hot Encoding)
    # Criar dummies usando a coluna original MTRANS_Original
    df_model = pd.get_dummies(df, columns=['MTRANS_Original'], prefix='Transporte', drop_first=True)
    
    # 8. Definir X (features) e y (target)
    
    # Selecionar as colunas que o modelo espera
    X = df_model[[
        'Genero_Model', 'Idade', 'Altura', 'Peso', 'Histórico_Model', 
        'Alimentos_Caloricos_Model', 'Vegetais?', 'Refeicoes_principais', 
        'Alimentos_entre_refeicoes?_Model', 'Fuma?_Model', 'Qtde_Agua', 
        'Monitora_calorias?_Model', 'Atividade_Fisica', 'Tempo_tecnologia', 
        'Alcool?_Model', 'IMC', 
        'Transporte_Bike', 'Transporte_Motorbike', 
        'Transporte_Public_Transportation', 'Transporte_Walking'
    ]]
    
    # Renomear as colunas X para o formato amigável usado na previsão
    X.columns = ['Genero', 'Idade', 'Altura', 'Peso', 'Histórico', 'Alimentos_Caloricos', 
                 'Vegetais?', 'Refeicoes_principais', 'Alimentos_entre_refeicoes?', 'Fuma?', 
                 'Qtde_Agua', 'Monitora_calorias?', 'Atividade_Fisica', 'Tempo_tecnologia', 
                 'Alcool?', 'IMC', 'Transporte_Bike', 'Transporte_Motorbike', 
                 'Transporte_Public_Transportation', 'Transporte_Walking']

    y = df_model['Nivel_Obesidade']
    
    # 9. Separar dados em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 10. Scaling (Normalização)
    colunas_numericas = ['Idade', 'Altura', 'Peso', 'Vegetais?', 'Refeicoes_principais',
                         'Alimentos_entre_refeicoes?', 'Qtde_Agua', 'Atividade_Fisica',
                         'Tempo_tecnologia', 'Alcool?', 'IMC']

    scaler = StandardScaler()
    X_train[colunas_numericas] = scaler.fit_transform(X_train[colunas_numericas])

    #Patch 11 para teste
    # 11. Treinar Random Forest
    #modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    #modelo_rf.fit(X_train, y_train)

    # Cálculo da acurácia (apenas para exibição)
    #acuracia = accuracy_score(y_test, modelo_rf.predict(X_test.fillna(0))) # .fillna(0) para evitar NaNs

    #Patch novo do 11
    # 11. Treinar Random Forest
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    modelo_rf.fit(X_train, y_train)

    # ============================================================
    # ✅ Acurácia correta (com X_test escalado)
    # O modelo foi treinado com X_train escalado, então o X_test
    # também precisa ser escalado com o MESMO scaler antes do predict.
    # ============================================================

    # --- Versão antiga (deixada aqui para voltar rápido se precisar) ---
    # acuracia = accuracy_score(y_test, modelo_rf.predict(X_test.fillna(0)))  # pode ficar inconsistente

    # --- Versão corrigida (recomendada) ---
    X_test_scaled = X_test.copy()
    X_test_scaled[colunas_numericas] = scaler.transform(X_test_scaled[colunas_numericas])
    acuracia = accuracy_score(y_test, modelo_rf.predict(X_test_scaled.fillna(0)))  # .fillna(0) para evitar NaNs

    
    # Variáveis Globais de Retorno para o Streamlit
    return df, modelo_rf, scaler, X_train.columns.tolist(), colunas_numericas, acuracia


# Carregar e treinar o modelo
df, modelo_rf, scaler, feature_cols, colunas_numericas, acuracia = load_and_train_model()

# ===================================================================
# FUNÇÃO DE PREVISÃO
# ===================================================================

def prever_nivel_obesidade(dados_entrada):
    
    # Mapeamentos para Decodificação (De PT-BR para número)
    map_genero = {'Masculino': 1, 'Feminino': 0}
    map_sim_nao = {'Sim': 1, 'Não': 0}
    map_ordinal_input = {'Não': 0, 'As vezes': 1, 'Frequentemente': 2, 'Sempre': 3}
    
    # 1. ENCODING BINÁRIO E ORDINAL
    data = {
        'Genero': map_genero[dados_entrada['Genero']],
        'Idade': dados_entrada['Idade'],
        'Altura': dados_entrada['Altura'],
        'Peso': dados_entrada['Peso'],
        'Histórico': map_sim_nao[dados_entrada['Histórico']],
        'Alimentos_Caloricos': map_sim_nao[dados_entrada['Alimentos_Caloricos']],
        'Vegetais?': dados_entrada['Vegetais?'],
        'Refeicoes_principais': dados_entrada['Refeicoes_principais'],
        'Alimentos_entre_refeicoes?': map_ordinal_input[dados_entrada['Alimentos_entre_refeicoes?']],
        'Fuma?': map_sim_nao[dados_entrada['Fuma?']],
        'Qtde_Agua': dados_entrada['Qtde_Agua'],
        'Monitora_calorias?': map_sim_nao[dados_entrada['Monitora_calorias?']],
        'Atividade_Fisica': dados_entrada['Atividade_Fisica'],
        'Tempo_tecnologia': dados_entrada['Tempo_tecnologia'],
        'Alcool?': map_ordinal_input[dados_entrada['Alcool?']],
    }
    
    # 2. CALCULAR IMC
    data['IMC'] = data['Peso'] / (data['Altura'] ** 2)
    
    # 3. CRIAR DATAFRAME e ONE-HOT ENCODING (Transporte)
    input_df = pd.DataFrame([data])
    
    # Inicializar colunas dummy para transporte com 0
    for col in feature_cols:
        if col.startswith('Transporte_'):
            input_df[col] = 0
            
    map_transp_to_dummy = {
    'Transporte publico': 'Transporte_Public_Transportation',
    'Caminhando': 'Transporte_Walking',
    'Motocicleta': 'Transporte_Motorbike',
    'Bicicleta': 'Transporte_Bike',
    'Veiculo': None  # baseline (fica tudo 0)
    }

    
    # Ativar a coluna dummy correta
    dummy_col_name = map_transp_to_dummy.get(dados_entrada['Transporte'])

    if dummy_col_name and dummy_col_name in feature_cols:
        input_df[dummy_col_name] = 1


    # 4. REORDENAR E FILTRAR COLUNAS
    input_df = input_df[feature_cols]

    # 5. SCALING (Normalização)
    input_df[colunas_numericas] = scaler.transform(input_df[colunas_numericas])

    # 6. PREDIÇÃO
    previsao_codificada = modelo_rf.predict(input_df)
    
    return previsao_codificada[0]


# ===================================================================
# SIDEBAR: INTERFACE DE PREVISÃO (ORGANIZAÇÃO APRIMORADA)
# ===================================================================

# --- Texto fixo antigo (caso queiram voltar) ---
# st.sidebar.markdown(f"*(Modelo treinado com 99,05% de acurácia)*")

# --- Texto dinâmico (mostra a acurácia calculada) ---
st.sidebar.markdown(f"*(Modelo treinado com {acuracia*100:.2f}% de acurácia)*")
st.sidebar.markdown("Preencha os dados abaixo e clique em 'Fazer Previsão' na área principal.")
st.sidebar.markdown(f"*(Modelo treinado com 99,05% de acurácia)*")

# --- UI INPUTS ---
with st.sidebar:
    
    st.markdown("### Dados Pessoais")
    genero = st.selectbox('Gênero:', ['Masculino', 'Feminino'])
    idade = st.slider('Idade (anos):', min_value=15, max_value=65, value=25)
    altura = st.number_input('Altura (m):', min_value=1.00, max_value=2.50, value=1.70, step=0.01)
    peso = st.number_input('Peso (kg):', min_value=30.0, max_value=200.0, value=75.0, step=0.1)
    historico = st.selectbox('Histórico Familiar de Obesidade:', ['Sim', 'Não'])

    st.markdown("### Hábitos Alimentares")
    alimentos_caloricos = st.selectbox('Consome Alimentos com Alto Teor Calórico (FAVC)?', ['Sim', 'Não'])
    vegetais = st.select_slider('Frequência de Consumo de Vegetais (FCVC):', options=[1, 2, 3], value=2, help="1=Pouco, 3=Muito")
    refeicoes_principais = st.select_slider('Refeições Principais/dia (NCP):', options=[1, 2, 3, 4], value=3)
    alimentos_entre_refeicoes = st.selectbox('Frequência de Alimentos entre Refeições (CAEC):', ['Não', 'As vezes', 'Frequentemente', 'Sempre'])

    st.markdown("### Estilo de Vida")
    fuma = st.selectbox('Fuma (SMOKE)?', ['Sim', 'Não'])
    qtde_agua = st.select_slider('Quantidade de Água (Qtde_Agua, L/dia):', options=[1, 2, 3], value=2)
    monitora_calorias = st.selectbox('Monitora calorias (SCC)?', ['Sim', 'Não'])
    atividade_fisica = st.select_slider('Frequência de Atividade Física (FAF):', options=[0, 1, 2, 3], value=1, help="0=Nenhum, 3=Sempre")
    tempo_tecnologia = st.select_slider('Tempo de Uso de Tecnologia (TUE):', options=[0, 1, 2], value=1, help="0=Pouco, 2=Muito")
    alcool = st.selectbox('Consumo de Álcool (CALC):', ['Não', 'As vezes', 'Frequentemente', 'Sempre'])
    
    transporte_opcoes = ['Transporte publico', 'Caminhando', 'Veiculo', 'Motocicleta', 'Bicicleta']
    transporte = st.selectbox('Principal Meio de Transporte:', transporte_opcoes)


# ===================================================================
# PÁGINA PRINCIPAL: MÉTRICAS E RESULTADO
# ===================================================================

# --- SEÇÃO 1: MÉTRICAS E ANÁLISE BÁSICA ---
st.header("Análise Média Geral")
# URLs das Imagens (Ícones)
img_homem = "https://cdn-icons-png.flaticon.com/128/3135/3135715.png"
img_mulher = "https://cdn-icons-png.flaticon.com/128/949/949635.png"

# Média geral para exibição
idade_g = df['Idade'].mean()
altura_g = df['Altura'].mean()
peso_g = df['Peso'].mean()
imc_g = df['IMC'].mean()

with st.container(border=True):
    col_img, col_metrics = st.columns([1, 4]) 
    with col_img:
        st.image("https://cdn-icons-png.flaticon.com/128/1256/1256650.png", width=120) # Ícone de gráfico/relatório
    with col_metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Idade Média", f"{idade_g:.0f} anos") 
        m2.metric("Altura Média", f"{altura_g:.2f} m")
        m3.metric("Peso Médio", f"{peso_g:.1f} kg")
        m4.metric("IMC Médio", f"{imc_g:.1f}")

st.write("---")

# --- SEÇÃO 2: RESULTADO DA PREVISÃO (Corpo Principal) ---
st.header("Resultados do Modelo de Previsão")
st.markdown("Configure as características do indivíduo na **Barra Lateral** (à esquerda) e utilize o botão abaixo para gerar a previsão.")

# Bloco para exibir o resultado da previsão
with st.container(border=True):
    
    col_btn, col_res = st.columns([1, 3])
    
    with col_btn:
        st.write("") # Espaçamento
        st.write("") # Espaçamento
        # --- BOTÃO ---
        if st.button('Fazer Previsão', use_container_width=True):
            
            # Coletar todos os inputs do Sidebar (eles já estão armazenados nas variáveis globais)
            user_inputs = {
                'Genero': genero, 'Idade': idade, 'Altura': altura, 'Peso': peso,
                'Histórico': historico, 'Alimentos_Caloricos': alimentos_caloricos,
                'Vegetais?': vegetais, 'Refeicoes_principais': refeicoes_principais,
                'Alimentos_entre_refeicoes?': alimentos_entre_refeicoes, 'Fuma?': fuma,
                'Qtde_Agua': qtde_agua, 'Monitora_calorias?': monitora_calorias,
                'Atividade_Fisica': atividade_fisica, 'Tempo_tecnologia': tempo_tecnologia,
                'Alcool?': alcool, 'Transporte': transporte
            }
            
            try:
                resultado_previsao = prever_nivel_obesidade(user_inputs)
                
                # Definir cor de destaque
                if 'Obesidade' in resultado_previsao:
                    cor = 'red'
                elif 'Sobrepeso' in resultado_previsao:
                    cor = 'orange'
                elif 'Peso Insuficiente' in resultado_previsao:
                    cor = '#555555' # Cinza
                else:
                    cor = 'green'
                
                # Armazenar o resultado na sessão para exibição
                st.session_state.previsao = resultado_previsao
                st.session_state.cor = cor
                st.session_state.fez_previsao = True

            except Exception as e:
                st.error("Erro ao processar a previsão. Verifique os valores de entrada.")
                # st.exception(e) # Descomente para debug
                st.session_state.fez_previsao = False
    
    # Exibir o resultado
    with col_res:
        if 'fez_previsao' in st.session_state and st.session_state.fez_previsao:
            st.markdown(f"**Nível de Peso Previsto:**")
            st.markdown(f"<h1 style='color: {st.session_state.cor}; font-size: 40px;'>{st.session_state.previsao}</h1>", unsafe_allow_html=True)
        else:
            st.info("Aguardando entrada de dados e clique no botão 'Fazer Previsão' na coluna ao lado.")