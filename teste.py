# Permutation importance (rápido)
from sklearn.inspection import permutation_importance
import joblib, pandas as pd
m = joblib.load('models/multi_no_bmi.joblib')
df_test = pd.read_csv('models/test_set.csv')
X_test = df_test.drop(columns=['Obesity'])
y_test = df_test['Obesity']
r = permutation_importance(m, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
imp = sorted(zip(X_test.columns, r.importances_mean), key=lambda x: -x[1])
print("Top features:", imp[:15])
