# 🧠 Previsão de Obesidade com Machine Learning

Este projeto faz parte do **Tech Challenge – Fase 4 (FIAP)** e tem como objetivo aplicar **Análise de Dados** e **Machine Learning** para prever o **nível de obesidade** de um indivíduo a partir de informações pessoais, hábitos alimentares e estilo de vida.

O resultado final é um **aplicativo interativo desenvolvido em Streamlit**, disponível online para testes.

---

## 🚀 Aplicação Online
🔗 **Link do App:**  
https://tech-challenge4-cny6yal8bsaikawuwe4ct.streamlit.app

> Obs.: No primeiro acesso, o carregamento pode levar alguns segundos (cold start do Streamlit).

---

## 📊 O que o app faz
- Treina um **modelo de Machine Learning (Random Forest)** a partir do dataset de obesidade
- Calcula métricas médias da base (idade, altura, peso e IMC)
- Permite ao usuário inserir seus próprios dados
- Retorna o **nível de obesidade previsto** pelo modelo
- Exibe o resultado de forma visual e intuitiva

---

## 🧠 Modelo de Machine Learning
- Algoritmo: **Random Forest Classifier**
- Features utilizadas:
  - Dados pessoais (idade, gênero, altura, peso)
  - Hábitos alimentares
  - Estilo de vida
  - IMC (calculado automaticamente)
- Estratégia de deploy: **treinamento no start da aplicação**
- Acurácia aproximada: **~99%** (validação treino/teste)

> O valor de acurácia é apresentado apenas como referência acadêmica.

---

## 🗂 Estrutura do Projeto
