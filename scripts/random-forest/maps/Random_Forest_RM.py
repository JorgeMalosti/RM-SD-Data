import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Definir nome do mapa e do parâmetro
mapa = "Tent"        # Exemplo: "Betax", "Lyapunov", etc.
parametro = "a"       # Nome da coluna do parâmetro no arquivo (pode ser qualquer string)

# Carregar o arquivo no formato: parametro P0 P1 ... P15
file_path = f"Data_{mapa}_RM.dat"
data = np.loadtxt(file_path)

# Criar o DataFrame com os nomes corretos das colunas
df = pd.DataFrame(data, columns=[parametro] + [f"Probs_{i}" for i in range(16)])

# Converter o valor contínuo do parâmetro em rótulos discretos
unique_vals = np.sort(df[parametro].unique())

# Verifica se existem exatamente 10 valores únicos
if len(unique_vals) != 10:
    print(f"Atenção: Esperava 10 valores únicos de {parametro}, mas foram encontrados {len(unique_vals)}.")

# Mapeia cada valor único do parâmetro para uma classe (0 a 9)
param_to_class = {val: i for i, val in enumerate(unique_vals)}
df["Param_class"] = df[parametro].map(param_to_class)

print(f"Mapping de {parametro} para classes:", param_to_class)

# Separar as variáveis independentes e a variável dependente
prob_cols = [f'Probs_{i}' for i in range(16)]
X = df[prob_cols].copy()
y = df["Param_class"]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Treinar o modelo Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Realizar previsões
y_pred = clf.predict(X_test)

# Calcular a acurácia
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.4f}")

# Gerar a matriz de confusão
conf_matrix = metrics.confusion_matrix(y_test, y_pred)

# Plotar e salvar a matriz de confusão como JPEG
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=60)
plt.title(f"Confusion Matrix - {mapa} Map")
plt.colorbar()
tick_marks = np.arange(len(unique_vals))
plt.xticks(tick_marks, [f"{val:.2f}" for val in unique_vals], rotation=45)
plt.yticks(tick_marks, [f"{val:.2f}" for val in unique_vals])
plt.xlabel("Predicted")
plt.ylabel("Real")

# Adicionar os valores nas células
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig(f"ConfusionMatrix_{mapa}_RM.jpeg", format="jpeg")

# Salvar a acurácia em uma nova linha do arquivo (modo append)
with open("Accuracies.txt", "a") as f:
    f.write("Accuracies:\n") if f.tell() == 0 else None  # Escreve o título só se o arquivo estiver vazio
    f.write(f"{mapa} = {accuracy:.4f}\n")

#plt.figure(figsize=(10, 8))
#sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_values_formatted, yticklabels=unique_values_formatted)
#plt.xlabel("Predicted")
#plt.ylabel("Real")
#plt.title("Confusion Matrix - SD - Lozi Map")
#plt.xticks(rotation=90)
#plt.yticks(rotation=0)
#plt.savefig("ConfusionMatrix_SD_Lozi.png")