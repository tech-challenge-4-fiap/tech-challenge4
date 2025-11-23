import streamlit as st
import altair as alt
#  py -m streamlit run meu_app.py
#  

# Este título fica na página principal
st.markdown("<h1 style='text-align: center;'>Relatório de analise sobre Obesidade</h1>", unsafe_allow_html=True)

st.write("---")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Ignorar warnings para uma saída mais limpa
warnings.filterwarnings('ignore')

# 1. Carregar Dados

# Tenta carregar o CSV. Certifique-se que 'Obesity.csv' está no local correto.
df = pd.read_csv('C:\\Users\\jonasl\\OneDrive - Pandurata Alimentos Ltda\\Documentos\\Obesity.csv')

# 2. Renomear Colunas
df.rename(columns={
    'Gender': 'Genero', 'Age': 'Idade', 'Height': 'Altura', 'Weight': 'Peso',
    'family_history': 'Histórico', 'SMOKE': 'Fuma?', 'FAF': 'Atividade_Fisica',
    'CALC': 'Alcool?', 'FAVC': 'Alimentos_Caloricos', 'FCVC': 'Vegetais?',
    'NCP': 'Refeicoes_principais', 'CAEC': 'Alimentos_entre_refeicoes?',
    'CH2O': 'Qtde_Agua', 'SCC': 'Monitora_calorias?', 'TUE': 'Tempo_tecnologia',
    'MTRANS': 'Transporte', 'Obesity': 'Nivel_Obesidade'
}, inplace=True)

# 3. Mapear Binários (sim/não, Male/Female)
rename_binary = {'yes': 'Sim', 'no': 'Não'}
rename_binary_genero = {'Male': 'Masculino', 'Female': 'Feminino'}
df['Genero'] = df['Genero'].map(rename_binary_genero)
df['Histórico'] = df['Histórico'].map(rename_binary)
df['Alimentos_Caloricos'] = df['Alimentos_Caloricos'].map(rename_binary)
df['Fuma?'] = df['Fuma?'].map(rename_binary)
df['Monitora_calorias?'] = df['Monitora_calorias?'].map(rename_binary)

# 4. Arredondamentos
df['Idade'] = df['Idade'].round(0)
df['Idade'] = df['Idade'].astype(int)
df['Altura'] = df['Altura'].round(2)
df['Peso'] = df['Peso'].round(1)
df['Refeicoes_principais'] = df['Refeicoes_principais'].round(0)
df['Qtde_Agua'] = df['Qtde_Agua'].round(0)
df['Atividade_Fisica'] = df['Atividade_Fisica'].round(0)
df['Tempo_tecnologia'] = df['Tempo_tecnologia'].round(0)

# 5. Feature Engineering (Criação do IMC)
df['IMC'] = df['Peso'] / (df['Altura'] * df['Altura'])

# 6. Mapear colunas para PT-BR
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
    'Public_Transportation': 'Transporte publico', 'Walking': 'Caminhando', 'Automobile': 'Veiculo', 'Motorbike': 'Motocicleta',
       'Bike':'Bicicleta'
}

df['Alimentos_entre_refeicoes?'] = df['Alimentos_entre_refeicoes?'].map(rename_colunas)
df['Alcool?'] = df['Alcool?'].map(rename_colunas)
df['Transporte'] = df['Transporte'].map(rename_transp)

grupo_peso = { 'Peso Normal': 'Peso Normal', 'Sobrepeso I':'Sobrepeso', 'Sobrepeso II': 'Sobrepeso', 'Obesidade I': 'Obesidade',
       'Peso Abaixo': 'Peso Abaixo', 'Obesidade II': 'Obesidade', 'Obesidade III': 'Obesidade',}

df['Grupo_Peso'] = df['Nivel_Obesidade'].map(grupo_peso)

# --------------------------------------- PRIMEIRA ETAPA (QUADRINHOS) -------------------------------------------------------------

# --- URLs das Imagens (Ícones) ---
img_homem = "https://cdn-icons-png.flaticon.com/128/3135/3135715.png"
img_mulher = "https://cdn-icons-png.flaticon.com/128/949/949635.png"

# ===================================================================
# LINHA DE ANÁLISE 1: MASCULINO
# ===================================================================
st.header("Análise Masculina")

with st.container(border=True):
    # 1. Filtrar dados
    df_m = df[df['Genero'] == 'Masculino']
    
    if not df_m.empty:
        # 2. Calcular Métricas
        idade_m = df_m['Idade'].mean()
        altura_m = df_m['Altura'].mean()
        peso_m = df_m['Peso'].mean()
        
        # --- CORREÇÃO AQUI ---
        # Adicionar .mean() para pegar a média da coluna IMC
        imc_m = df_m['IMC'].mean() 

        # 3. Criar Layout (Imagem + Bloco de Métricas)
        col_img, col_metrics = st.columns([1, 4]) 

        with col_img:
            st.image(img_homem, width=120)

        with col_metrics:
            m1, m2, m3, m4 = st.columns(4)
            # Você também mudou a formatação da idade, o que é ótimo!
            m1.metric("Idade Média", f"{idade_m:.0f} anos") 
            m2.metric("Altura Média", f"{altura_m:.2f} m")
            m3.metric("Peso Médio", f"{peso_m:.1f} kg")
            m4.metric("IMC Médio", f"{imc_m:.1f}")
            
    else:
        st.warning("Nenhum dado 'Masculino' encontrado para análise.")

st.write(" ") # Adiciona um espaço vertical

# ===================================================================
# LINHA DE ANÁLISE 2: FEMININO
# ===================================================================
st.header("Análise Feminina")

with st.container(border=True):
    # 1. Filtrar dados
    df_f = df[df['Genero'] == 'Feminino']
    
    if not df_f.empty:
        # 2. Calcular Métricas
        idade_f = df_f['Idade'].mean()
        altura_f = df_f['Altura'].mean()
        peso_f = df_f['Peso'].mean()
        imc_f = df_f['IMC'].mean()

        # 3. Criar Layout (Imagem + Bloco de Métricas)
        col_img, col_metrics = st.columns([1, 4]) # Imagem (1 parte) | Métricas (4 partes)

        with col_img:
            st.image(img_mulher, width=120)

        with col_metrics:
            # st.subheader("Resumo Feminino") # Título opcional dentro do card

            # Criar 4 colunas para as 4 métricas
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Idade Média", f"{idade_f:.0f} anos")
            m2.metric("Altura Média", f"{altura_f:.2f} m")
            m3.metric("Peso Médio", f"{peso_f:.1f} kg")
            m4.metric("IMC Médio", f"{imc_f:.1f}")
            
    else:
        st.warning("Nenhum dado 'Feminino' encontrado para análise.")


        st.write("---")

import altair as alt
import pandas as pd

import altair as alt
import pandas as pd


st.write("---")

# ===================================================================
# NOVA SEÇÃO: GRÁFICO DE BARRAS EMPILHADAS (100%)
# ===================================================================
st.header("Nivel de Obesidade por Gênero")

    
# --- ETAPA DE PREPARAÇÃO DE DADOS (NECESSÁRIA PARA RÓTULOS) ---

# 1. Preparar os dados: Contar as combinações
df_counts = df.groupby(['Nivel_Obesidade', 'Genero']).size().reset_index(name='Contagem')

# 2. Calcular o total para CADA Nivel_Obesidade
total_por_nivel = df_counts.groupby('Nivel_Obesidade')['Contagem'].transform('sum')

# 3. Calcular a coluna de percentual
df_counts['Percentual'] = df_counts['Contagem'] / total_por_nivel

# --- FIM DA PREPARAÇÃO ---


# 4. Criar o gráfico base
# O X e Y são definidos aqui. O empilhamento é inferido pelo 'color'.
base = alt.Chart(df_counts).encode(
    
    # X: Nivel_Obesidade (Eixo categórico)
    x=alt.X('Nivel_Obesidade', title='Nível de Obesidade', sort=None),
    
    # Y: Usar a coluna 'Percentual' que calculamos
    y=alt.Y('Percentual', title='Percentual (%)', axis=alt.Axis(format='%')),
    
    # Cor: A legenda (Gênero) com escala de cores customizada
    color=alt.Color('Genero', 
                    title='Gênero',
                    scale=alt.Scale(
                        domain=['Feminino', 'Masculino'],
                        range=['lightpink', 'darkblue'] # <-- REQUISITO 1: Cores
                    )),
    
    # Tooltip para interatividade
    tooltip=[
        'Nivel_Obesidade', 
        'Genero', 
        'Contagem',
        alt.Tooltip('Percentual', format='.1%') # Adiciona o percentual formatado
    ]
)

# 5. Criar a camada de barras
bars = base.mark_bar()

# 7. Combinar as camadas de barras e texto
# Isso agora funciona porque 'base' não contém 'column' (faceting)
chart = (bars).interactive()

# 8. Exibir o gráfico no Streamlit
st.altair_chart(chart, use_container_width=True)



# ===================================================================
# 2º: GRÁFICO DE BARRAS EMPILHADAS (100%)
# ===================================================================
st.header("Histórico por Grupo de Peso")

# 1. Preparar os dados: Contar as combinações
df_counts = df.groupby(['Grupo_Peso', 'Histórico']).size().reset_index(name='Contagem')

# 2. Calcular o total para CADA Grupo_Peso
total_por_nivel = df_counts.groupby('Grupo_Peso')['Contagem'].transform('sum')

# 3. Calcular a coluna de percentual
df_counts['Percentual'] = df_counts['Contagem'] / total_por_nivel

# 4. Criar o gráfico base
# O X e Y são definidos aqui. O empilhamento é inferido pelo 'color'.
base = alt.Chart(df_counts).encode(
    
    # X: Grupo_Peso (Eixo categórico)
    x=alt.X('Grupo_Peso', title='Grupo de Peso', sort=None), # Título corrigido para 'Grupo de Peso'
    
    # Y: Usar a coluna 'Percentual' que calculamos
    y=alt.Y('Percentual', title='Percentual (%)', axis=alt.Axis(format='%')),
    
    # --- CORREÇÃO ---
    # Cor: A legenda (stack) deve ser 'Histórico', não 'Genero'
    color=alt.Color('Histórico', 
                    title='Histórico'
                    # A escala de cores personalizada foi removida pois era para 'Genero'
                    ),
    
    # Tooltip para interatividade
    tooltip=[
        'Grupo_Peso', 
        'Histórico', # <-- CORRIGIDO (era 'Genero')
        'Contagem',
        alt.Tooltip('Percentual', format='.1%') # Adiciona o percentual formatado
    ]
)

# 5. Criar a camada de barras
bars = base.mark_bar()

# 7. Combinar as camadas de barras e texto
# (O usuário removeu a camada de texto, então apenas 'bars' é necessário)
chart = (bars).interactive()

# 8. Exibir o gráfico no Streamlit
st.altair_chart(chart, use_container_width=True)