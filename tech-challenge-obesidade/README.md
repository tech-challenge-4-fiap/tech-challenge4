# 🧠 Previsão e Análise de Obesidade com Machine Learning

Este projeto faz parte do **Tech Challenge – Fase 4 (FIAP)** e tem como objetivo aplicar **Análise de Dados** e **Machine Learning** para estudar e prever o **nível de obesidade** de indivíduos com base em dados pessoais, hábitos alimentares e estilo de vida.

O trabalho foi dividido em **duas aplicações online**, ambas desenvolvidas em **Streamlit**:

---

## 🚀 Aplicações Online

### 🔮 1. App de Previsão de Obesidade (Machine Learning)
Aplicação interativa onde o usuário informa seus dados e o modelo de Machine Learning retorna o **nível de obesidade previsto**.

🔗 **Link do App de Previsão:**  
https://tech-challenge4-cny6yal8bsaikawuwe4ct.streamlit.app

**Funcionalidades:**
- Entrada de dados via sidebar
- Cálculo automático do IMC
- Previsão do nível de obesidade
- Exibição visual do resultado
- Modelo treinado em tempo de execução

---

### 📊 2. Dashboard Analítico de Obesidade
Dashboard focado na **análise exploratória dos dados**, com métricas e gráficos interativos que ajudam a entender padrões de obesidade na base utilizada.

🔗 **Link do Dashboard:**  
https://tech-challenge4-zkhtkv39u54mpdywugkjds.streamlit.app/

**Análises disponíveis:**
- Comparação entre gêneros (masculino x feminino)
- Distribuição do nível de obesidade por gênero
- Relação entre histórico familiar e grupo de peso
- Média de atividade física por grupo de peso
- Métricas médias (idade, altura, peso e IMC)

---

## 🧠 Modelo de Machine Learning
- Algoritmo: **Random Forest Classifier**
- Features:
  - Dados pessoais (idade, gênero, altura, peso)
  - Hábitos alimentares
  - Estilo de vida
  - IMC (calculado automaticamente)
- Estratégia de deploy: **treinamento no start da aplicação**
- Acurácia aproximada: **~99%** (validação treino/teste)

> O modelo tem finalidade **educacional** e não substitui avaliação médica.

---

## 🗂 Estrutura do Projeto

tech-challenge-obesidade/
│
├── app/
│ └── app.py # App de previsão com ML
│
├── dashboard/
│ └── dashboard.py # Dashboard analítico
│
├── data/
│ └── Obesity.csv # Dataset utilizado
│
├── .streamlit/
│ └── config.toml # Configuração de tema
│
├── README.md
└── requirements.txt


---

## 🛠 Tecnologias Utilizadas
- Python
- Pandas
- NumPy
- Scikit-learn
- Altair
- Streamlit
- Git & GitHub
- Streamlit Cloud

---

## 👥 Projeto em Grupo
Projeto desenvolvido em grupo como parte do **Tech Challenge – FIAP**, com foco em:
- Análise exploratória de dados
- Modelagem preditiva
- Visualização de informações
- Deploy de aplicações de dados

---

## ⚠️ Observações
- Os resultados são baseados no dataset utilizado
- O projeto possui **finalidade acadêmica**
