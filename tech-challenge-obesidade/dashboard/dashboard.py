import streamlit as st
import altair as alt
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG + CSS
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="Relatório de Obesidade",
    page_icon="📊"
)

st.markdown("""
<style>
.stApp { background-color: #f0f2f6; }
h1 { color: #1f77b4; }
h2 { border-bottom: 2px solid #e1e4e8; padding-bottom: 5px; margin-top: 25px; color: #333333; }
.stContainer {
    background-color: white !important;
    border-radius: 10px;
    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.1);
    padding: 20px;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e6e6e6;
    box-shadow: 0 2px 4px 0 rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ORDENS
order_nivel_obesidade = ['Peso Abaixo', 'Peso Normal', 'Sobrepeso I', 'Sobrepeso II', 'Obesidade I', 'Obesidade II', 'Obesidade III']
order_grupo_peso = ['Peso Abaixo', 'Peso Normal', 'Sobrepeso', 'Obesidade']

st.markdown("<h1 style='text-align: center;'>Relatório de análise sobre Obesidade</h1>", unsafe_allow_html=True)
st.write("---")

# ============================================================
# CARREGAMENTO DE DADOS (DEPLOY FRIENDLY)
# ============================================================
@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR.parent / "data" / "Obesity.csv"

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {DATA_PATH}. Coloque o Obesity.csv em tech-challenge-obesidade/data/")
        st.stop()

    # Renomear colunas
    df.rename(columns={
        'Gender': 'Genero', 'Age': 'Idade', 'Height': 'Altura', 'Weight': 'Peso',
        'family_history': 'Histórico', 'SMOKE': 'Fuma?', 'FAF': 'Atividade_Fisica',
        'CALC': 'Alcool?', 'FAVC': 'Alimentos_Caloricos', 'FCVC': 'Vegetais?',
        'NCP': 'Refeicoes_principais', 'CAEC': 'Alimentos_entre_refeicoes?',
        'CH2O': 'Qtde_Agua', 'SCC': 'Monitora_calorias?', 'TUE': 'Tempo_tecnologia',
        'MTRANS': 'Transporte', 'Obesity': 'Nivel_Obesidade'
    }, inplace=True)

    # Mapear binários
    rename_binary = {'yes': 'Sim', 'no': 'Não'}
    rename_binary_genero = {'Male': 'Masculino', 'Female': 'Feminino'}

    df['Genero'] = df['Genero'].map(rename_binary_genero)
    df['Histórico'] = df['Histórico'].map(rename_binary)
    df['Alimentos_Caloricos'] = df['Alimentos_Caloricos'].map(rename_binary)
    df['Fuma?'] = df['Fuma?'].map(rename_binary)
    df['Monitora_calorias?'] = df['Monitora_calorias?'].map(rename_binary)

    # Arredondamentos
    df['Idade'] = df['Idade'].round(0).astype(int)
    df['Altura'] = df['Altura'].round(2)
    df['Peso'] = df['Peso'].round(1)
    df['Refeicoes_principais'] = df['Refeicoes_principais'].round(0).astype(int)
    df['Qtde_Agua'] = df['Qtde_Agua'].round(0).astype(int)
    df['Atividade_Fisica'] = df['Atividade_Fisica'].round(0).astype(int)
    df['Tempo_tecnologia'] = df['Tempo_tecnologia'].round(0).astype(int)

    # IMC
    df['IMC'] = df['Peso'] / (df['Altura'] ** 2)

    # PT-BR target
    rename_target = {
        'Normal_Weight': 'Peso Normal', 'Overweight_Level_I': 'Sobrepeso I',
        'Overweight_Level_II': 'Sobrepeso II', 'Obesity_Type_I': 'Obesidade I',
        'Insufficient_Weight': 'Peso Abaixo', 'Obesity_Type_II': 'Obesidade II',
        'Obesity_Type_III': 'Obesidade III'
    }
    df['Nivel_Obesidade'] = df['Nivel_Obesidade'].map(rename_target)

    rename_colunas = {
        'Sometimes':'As vezes', 'Frequently':'Frequentemente', 'Always':'Sempre', 'no':'Não'
    }
    rename_transp = {
        'Public_Transportation': 'Transporte publico', 'Walking': 'Caminhando',
        'Automobile': 'Veiculo', 'Motorbike': 'Motocicleta', 'Bike':'Bicicleta'
    }

    df['Alimentos_entre_refeicoes?'] = df['Alimentos_entre_refeicoes?'].map(rename_colunas).fillna(df['Alimentos_entre_refeicoes?'])
    df['Alcool?'] = df['Alcool?'].map(rename_colunas).fillna(df['Alcool?'])
    df['Transporte'] = df['Transporte'].map(rename_transp).fillna(df['Transporte'])

    # Grupo peso
    grupo_peso = {
        'Peso Normal': 'Peso Normal',
        'Sobrepeso I': 'Sobrepeso',
        'Sobrepeso II': 'Sobrepeso',
        'Obesidade I': 'Obesidade',
        'Obesidade II': 'Obesidade',
        'Obesidade III': 'Obesidade',
        'Peso Abaixo': 'Peso Abaixo'
    }
    df['Grupo_Peso'] = df['Nivel_Obesidade'].map(grupo_peso)

    return df

df = load_data()

# ============================================================
# CARDS MASCULINO/FEMININO
# ============================================================
img_homem = "https://cdn-icons-png.flaticon.com/128/3135/3135715.png"
img_mulher = "https://cdn-icons-png.flaticon.com/128/949/949635.png"

st.header("Análise Masculina")
with st.container(border=True):
    df_m = df[df['Genero'] == 'Masculino']
    if df_m.empty:
        st.warning("Nenhum dado masculino encontrado.")
    else:
        col_img, col_metrics = st.columns([1, 4])
        with col_img:
            st.image(img_homem, width=120)
        with col_metrics:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Idade Média", f"{df_m['Idade'].mean():.0f} anos")
            m2.metric("Altura Média", f"{df_m['Altura'].mean():.2f} m")
            m3.metric("Peso Médio", f"{df_m['Peso'].mean():.1f} kg")
            m4.metric("IMC Médio", f"{df_m['IMC'].mean():.1f}")

st.write(" ")

st.header("Análise Feminina")
with st.container(border=True):
    df_f = df[df['Genero'] == 'Feminino']
    if df_f.empty:
        st.warning("Nenhum dado feminino encontrado.")
    else:
        col_img, col_metrics = st.columns([1, 4])
        with col_img:
            st.image(img_mulher, width=120)
        with col_metrics:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Idade Média", f"{df_f['Idade'].mean():.0f} anos")
            m2.metric("Altura Média", f"{df_f['Altura'].mean():.2f} m")
            m3.metric("Peso Médio", f"{df_f['Peso'].mean():.1f} kg")
            m4.metric("IMC Médio", f"{df_f['IMC'].mean():.1f}")

st.write("---")

# ============================================================
# 1) Nível de obesidade por gênero (100% stacked)
# ============================================================
st.header("1. Nível de Obesidade por Gênero")

df_counts_1 = df.groupby(['Nivel_Obesidade', 'Genero']).size().reset_index(name='Contagem')
total_por_nivel_1 = df_counts_1.groupby('Nivel_Obesidade')['Contagem'].transform('sum')
df_counts_1['Percentual'] = df_counts_1['Contagem'] / total_por_nivel_1

chart_1 = alt.Chart(df_counts_1).mark_bar().encode(
    x=alt.X('Nivel_Obesidade:N', title='Nível de Obesidade', sort=order_nivel_obesidade),
    y=alt.Y('Percentual:Q', title='Percentual (%)', axis=alt.Axis(format='%')),
    color=alt.Color('Genero:N', title='Gênero',
        scale=alt.Scale(domain=['Feminino', 'Masculino'], range=['#e377c2', '#1f77b4'])
    ),
    tooltip=['Nivel_Obesidade', 'Genero', 'Contagem', alt.Tooltip('Percentual:Q', format='.1%')]
).properties(title='Distribuição de Gênero em Cada Nível de Obesidade').interactive()

st.altair_chart(chart_1, use_container_width=True)

# ============================================================
# 2) Histórico familiar por grupo de peso (100% stacked)
# ============================================================
st.header("2. Histórico Familiar por Grupo de Peso")

df_counts_2 = df.groupby(['Grupo_Peso', 'Histórico']).size().reset_index(name='Contagem')
total_por_nivel_2 = df_counts_2.groupby('Grupo_Peso')['Contagem'].transform('sum')
df_counts_2['Percentual'] = df_counts_2['Contagem'] / total_por_nivel_2

chart_2 = alt.Chart(df_counts_2).mark_bar().encode(
    x=alt.X('Grupo_Peso:N', title='Grupo de Peso', sort=order_grupo_peso),
    y=alt.Y('Percentual:Q', title='Percentual (%)', axis=alt.Axis(format='%')),
    color=alt.Color('Histórico:N', title='Histórico Familiar',
        scale=alt.Scale(domain=['Sim', 'Não'], range=['#d62728', '#2ca02c'])
    ),
    tooltip=['Grupo_Peso', 'Histórico', 'Contagem', alt.Tooltip('Percentual:Q', format='.1%')]
).properties(title='Impacto do Histórico Familiar no Grupo de Peso').interactive()

st.altair_chart(chart_2, use_container_width=True)

st.write("---")

# ============================================================
# 3) Média de atividade física por grupo de peso
# ============================================================
st.header("3. Média de Atividade Física por Grupo de Peso")

df_atividade = df.groupby('Grupo_Peso')['Atividade_Fisica'].mean().reset_index(name='Media_Atividade')

base = alt.Chart(df_atividade).encode(
    x=alt.X('Grupo_Peso:N', title='Grupo de Peso', sort=order_grupo_peso),
    y=alt.Y('Media_Atividade:Q', title='Média (0:Não, 3:Sempre)', scale=alt.Scale(domain=[0, 3])),
    color=alt.Color('Grupo_Peso:N', title='Grupo de Peso', sort=order_grupo_peso,
                    scale=alt.Scale(range=['#a1d99b', '#74c476', '#31a354', '#006d2c']))
)

bars = base.mark_bar().encode(
    tooltip=['Grupo_Peso', alt.Tooltip('Media_Atividade:Q', title='Média', format='.2f')]
)

text = base.mark_text(align='center', baseline='bottom', dy=-5).encode(
    text=alt.Text('Media_Atividade:Q', format='.2f'),
    color=alt.value('black')
)

chart_3 = (bars + text).properties(
    title='Média de Atividade Física (FAF) por Grupo de Peso'
).interactive()

st.altair_chart(chart_3, use_container_width=True)
