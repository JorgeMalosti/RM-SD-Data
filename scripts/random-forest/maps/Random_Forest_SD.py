import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Definir nome do mapa e do parâmetro
mapa = "Tent"        # Exemplo: "Betax", "Lyapunov", etc.
parametro = "μ"      # Nome do parâmetro (para rotulagem apenas)

# Carregar o arquivo com separador de tabulação
file_path = f"Data_{mapa}_SD.dat"
df_raw = pd.read_csv(file_path, sep="\t")

# Manter apenas as colunas necessárias
df = df_raw[[df_raw.columns[0], df_raw.columns[2]]].copy()
df.columns = [parametro, "Pattern_Probabilities"]

# Expandir a coluna de probabilidades em 16 colunas numéricas
probs_expanded = df["Pattern_Probabilities"].str.split(",", expand=True).astype(float)
probs_expanded.columns = [f"Probs_{i}" for i in range(16)]

# Concatenar com o parâmetro
df_final = pd.concat([df[[parametro]], probs_expanded], axis=1)

# Criar classes discretas a partir dos valores do parâmetro
unique_vals = np.sort(df_final[parametro].unique())

if len(unique_vals) != 10:
    print(f"Atenção: Esperava 10 valores únicos de {parametro}, mas foram encontrados {len(unique_vals)}.")

# Mapeia os valores únicos do parâmetro para classes (0 a 9)
param_to_class = {val: i for i, val in enumerate(unique_vals)}
df_final["Param_class"] = df_final[parametro].map(param_to_class)

print(f"Mapping de {parametro} para classes:", param_to_class)

# Preparar dados para o modelo
X = df_final[[f'Probs_{i}' for i in range(16)]].copy()
y = df_final["Param_class"]

# Dividir em treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Treinar modelo
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Previsões e acurácia
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.4f}")

# Matriz de confusão
conf_matrix = metrics.confusion_matrix(y_test, y_pred)

# Plotar e salvar matriz de confusão
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=60)
plt.title(f"Confusion Matrix - {mapa} Map - SD")
plt.colorbar()
tick_marks = np.arange(len(unique_vals))
plt.xticks(tick_marks, [f"{val:.2f}" for val in unique_vals], rotation=45)
plt.yticks(tick_marks, [f"{val:.2f}" for val in unique_vals])
plt.xlabel("Predicted")
plt.ylabel("Real")

# Adicionar valores nas células
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig(f"ConfusionMatrix_{mapa}_SD.jpeg", format="jpeg")

# Salvar a acurácia no arquivo (append)
with open("Accuracies.txt", "a") as f:
    if f.tell() == 0:
        f.write("Accuracies:\n")
    f.write(f"{mapa} = {accuracy:.4f}\n")
