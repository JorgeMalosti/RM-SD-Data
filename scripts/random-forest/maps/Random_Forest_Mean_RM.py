import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Definir nome do mapa e do parâmetro
mapa = "Tent"
parametro = "a"  # Nome da coluna do parâmetro no arquivo
num_files = 5    # Número de arquivos a serem processados

# Inicializar acumuladores
conf_matrix_total = np.zeros((10, 10), dtype=int)
accuracies = []

# Processar múltiplos arquivos
for i in range(1, num_files + 1):
    file_path = f"Data_{mapa}_RM_{i}.dat"
    print(f"Processando arquivo: {file_path}")

    # Carregar os dados
    data = np.loadtxt(file_path)
    df = pd.DataFrame(data, columns=[parametro] + [f"Probs_{j}" for j in range(16)])

    # Converter o valor contínuo do parâmetro em rótulos discretos
    unique_vals = np.sort(df[parametro].unique())

    if len(unique_vals) != 10:
        print(f"Atenção: Esperava 10 valores únicos de {parametro}, mas foram encontrados {len(unique_vals)}.")

    param_to_class = {val: idx for idx, val in enumerate(unique_vals)}
    df["Param_class"] = df[parametro].map(param_to_class)

    #print(f"Mapping de {parametro} para classes:", param_to_class)

    # Dados de entrada e saída
    X = df[[f"Probs_{j}" for j in range(16)]].copy()
    y = df["Param_class"]

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Modelo Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Previsões e métricas
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    #print(f"Acurácia do modelo {file_path}: {acc:.4f}")

    # Matriz de confusão
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    conf_matrix_total += conf_matrix

# Calcular a matriz média de confusão
conf_matrix_avg = conf_matrix_total / num_files

# Plotar matriz de confusão média
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_avg, interpolation='nearest', cmap=plt.cm.Reds, vmin=0, vmax=60)
plt.title(f"Average Confusion Matrix - {mapa} Map - RM")
plt.colorbar()
tick_marks = np.arange(len(unique_vals))
plt.xticks(tick_marks, [f"{val:.2f}" for val in unique_vals], rotation=45)
plt.yticks(tick_marks, [f"{val:.2f}" for val in unique_vals])
plt.xlabel("Predicted")
plt.ylabel("Real")

# Adicionar os valores nas células
thresh = conf_matrix_avg.max() / 2.
for i in range(conf_matrix_avg.shape[0]):
    for j in range(conf_matrix_avg.shape[1]):
        plt.text(j, i, f"{int(conf_matrix_avg[i, j])}",
                 horizontalalignment="center",
                 color="white" if conf_matrix_avg[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig(f"ConfusionMatrix_Average_{mapa}_RM.jpeg", format="jpeg")

# Salvar acurácia média
#mean_accuracy = np.mean(accuracies)
#with open("Accuracies_RM.txt", "a") as f:
#    if f.tell() == 0:
#        f.write("Accuracies:\n")
#    f.write(f"{mapa} = {mean_accuracy:.4f}\n")
