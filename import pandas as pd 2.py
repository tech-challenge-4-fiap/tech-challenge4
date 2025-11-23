import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # Importando o Random Forest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# Ignorar warnings para uma saída mais limpa
warnings.filterwarnings('ignore')

# 1. Carregar Dados

# Tenta carregar o CSV. Certifique-se que 'Obesity.csv' está no local correto.
df = pd.read_csv('')

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
rename_binary = {'yes': 1, 'no': 0}
rename_binary_genero = {'Male': 1, 'Female': 0}
df['Genero'] = df['Genero'].map(rename_binary_genero)
df['Histórico'] = df['Histórico'].map(rename_binary)
df['Alimentos_Caloricos'] = df['Alimentos_Caloricos'].map(rename_binary)
df['Fuma?'] = df['Fuma?'].map(rename_binary)
df['Monitora_calorias?'] = df['Monitora_calorias?'].map(rename_binary)

# 4. Arredondamentos
df['Qtde_Agua'] = df['Qtde_Agua'].round(1)
df['Idade'] = df['Idade'].round(0)
df['Altura'] = df['Altura'].round(2)
df['Peso'] = df['Peso'].round(1)
df['Vegetais?'] = df['Vegetais?'].round(0)
df['Refeicoes_principais'] = df['Refeicoes_principais'].round(0)
df['Qtde_Agua'] = df['Qtde_Agua'].round(0)
df['Atividade_Fisica'] = df['Atividade_Fisica'].round(0)
df['Tempo_tecnologia'] = df['Tempo_tecnologia'].round(0)

# 6. Mapear Target (Nivel_Obesidade) para PT-BR
rename_target = {
    'Normal_Weight': 'Peso Normal', 'Overweight_Level_I': 'Sobrepeso I',
    'Overweight_Level_II': 'Sobrepeso II', 'Obesity_Type_I': 'Obesidade I',
    'Insufficient_Weight': 'Peso Insuficiente', 'Obesity_Type_II': 'Obesidade II',
    'Obesity_Type_III': 'Obesidade III'
}
df['Nivel_Obesidade'] = df['Nivel_Obesidade'].map(rename_target)

# 7. ENCODING

map_ordinal = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}

# Usamos .get(x, 0) para tratar valores inesperados (como 'Unknown' ou NaN), mapeando-os para 0 (Não)
df['Alcool?'] = df['Alcool?'].apply(lambda x: map_ordinal.get(x, 0))
df['Alimentos_entre_refeicoes?'] = df['Alimentos_entre_refeicoes?'].apply(lambda x: map_ordinal.get(x, 0))

# 7b. Encoding Nominal (One-Hot Encoding)
# Cria novas colunas (dummies) para 'Transporte', pois não há ordem
# drop_first=True remove a primeira categoria para evitar multicolinearidade
df = pd.get_dummies(df, columns=['Transporte'], drop_first=True)

df = df.drop(columns=['Peso', 'Altura'])

# 8. Definir X (features) e y (target)
NOME_DA_COLUNA_ALVO = 'Nivel_Obesidade'

y = df[NOME_DA_COLUNA_ALVO]
X = df.drop(NOME_DA_COLUNA_ALVO, axis=1)

# 9. Separar dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Tamanho do treino: {X_train.shape[0]} amostras")
print(f"Tamanho do teste: {X_test.shape[0]} amostras")
print("\n")

# 10. Scaling (Normalização)
# Lista de colunas para escalar (excluímos as colunas binárias/dummy criadas)
colunas_numericas = ['Idade', 'Vegetais?', 'Refeicoes_principais',
                        'Alimentos_entre_refeicoes?', 'Qtde_Agua', 'Atividade_Fisica',
                        'Tempo_tecnologia', 'Alcool?']

scaler = StandardScaler()
# Ajustar (fit) o scaler APENAS nos dados de TREINO
X_train[colunas_numericas] = scaler.fit_transform(X_train[colunas_numericas])
# Aplicar (transform) o scaler nos dados de TESTE
X_test[colunas_numericas] = scaler.transform(X_test[colunas_numericas])

modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Treinar o modelo com os dados de treino (X_train, y_train)
modelo_rf.fit(X_train, y_train)

print("Modelo Random Forest treinado com sucesso!")
print("\n")

# 12. Fazer Predições com os dados de Teste
y_pred_rf = modelo_rf.predict(X_test)

# 13. Avaliar Acurácia
acuracia_rf = accuracy_score(y_test, y_pred_rf)
print(f"Acurácia (Random Forest): {acuracia_rf * 100:.2f}%")

df.head()