import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Caminho do dataset
DATASET_PATH = "Fakenews-dataset-final.csv"

# Carregando os dados
df = pd.read_csv(DATASET_PATH)

# Removendo espaços em branco e caracteres invisíveis nos nomes das colunas
df.columns = df.columns.str.strip()

# Separando variáveis preditoras e alvo
X = df.drop(columns=["Classe"])
y = df["Classe"]

# Dividindo os dados com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Treinando o modelo Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Avaliando o modelo
y_pred = rf.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Precisão:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1-Score:", f1_score(y_test, y_pred, zero_division=0))

# Análise com SHAP
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)


# Beeswarm plot
shap.summary_plot(shap_values[1], X_test, show=False)
plt.savefig("shap_beeswarm.png", bbox_inches="tight")

plt.savefig("shap_beeswarm.png", bbox_inches="tight")

shap.summary_plot(shap_values[1], X_test, show=False)
plt.savefig("shap_beeswarm.png", bbox_inches="tight")

# Dependency plots para as 5 features mais importantes
importances = np.abs(shap_values[1]).mean(axis=0)
top5_idx = importances.argsort()[::-1][:5]
top5_features = X_test.columns[top5_idx]

for feature in top5_features:
    shap.dependence_plot(feature, shap_values[1], X_test, show=False)
    plt.savefig(f"shap_dependence_{feature}.png", bbox_inches="tight")
