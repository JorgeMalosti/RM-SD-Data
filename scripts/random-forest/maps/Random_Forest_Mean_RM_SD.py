import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Nome do mapa e do parâmetro
mapa = "Tent"
parametro = "μ"
num_files = 5

# Acumuladores
conf_matrix_total = np.zeros((10, 10), dtype=int)
accuracies = []

# Loop pelos arquivos RM e SD
for i in range(1, num_files + 1):
    # --- MICROESTADOS (RM) ---
    file_rm = f"Data_{mapa}_RM_{i}.dat"
    print(f"Processando arquivo RM: {file_rm}")
    data_rm = np.loadtxt(file_rm)
    df_rm = pd.DataFrame(data_rm, columns=[parametro] + [f"Probs_RM_{j}" for j in range(16)])

    # --- DINÂMICA SIMBÓLICA (SD) ---
    file_sd = f"Data_{mapa}_SD_{i}.dat"
    print(f"Processando arquivo SD: {file_sd}")
    df_sd_raw = pd.read_csv(file_sd, sep="\t")
    df_sd = df_sd_raw[[df_sd_raw.columns[0], df_sd_raw.columns[2]]].copy()
    df_sd.columns = [parametro, "Pattern_Probabilities"]
    probs_sd = df_sd["Pattern_Probabilities"].str.split(",", expand=True).astype(float)
    probs_sd.columns = [f"Probs_SD_{j}" for j in range(16)]
    df_sd_final = pd.concat([df_sd[[parametro]], probs_sd], axis=1)

    # Verificar alinhamento das amostras
    if not np.allclose(df_rm[parametro].values, df_sd_final[parametro].values):
        raise ValueError(f"Parâmetros não coincidem entre RM e SD no arquivo {i}.")

    # Combinar RM + SD
    df_combined = pd.concat([df_rm[[parametro]], df_rm.drop(columns=[parametro]), df_sd_final.drop(columns=[parametro])], axis=1)

    # Criar classes discretas
    unique_vals = np.sort(df_combined[parametro].unique())
    if len(unique_vals) != 10:
        print(f"Atenção: Esperava 10 valores únicos de {parametro}, mas foram encontrados {len(unique_vals)}.")
    param_to_class = {val: idx for idx, val in enumerate(unique_vals)}
    df_combined["Param_class"] = df_combined[parametro].map(param_to_class)

    # Separar features e target
    X = df_combined[[col for col in df_combined.columns if "Probs_" in col]].copy()
    y = df_combined["Param_class"]

    # Treinamento/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Modelo
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Avaliação
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    conf_matrix_total += conf_matrix

# Média das matrizes de confusão
conf_matrix_avg = conf_matrix_total / num_files

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_avg, interpolation='nearest', cmap=plt.cm.Reds, vmin=0, vmax=60)
plt.title(f"Average Confusion Matrix - {mapa} - RM + SD")
plt.colorbar()
tick_marks = np.arange(len(unique_vals))
plt.xticks(tick_marks, [f"{val:.2f}" for val in unique_vals], rotation=45)
plt.yticks(tick_marks, [f"{val:.2f}" for val in unique_vals])
plt.xlabel("Predicted")
plt.ylabel("Real")

thresh = conf_matrix_avg.max() / 2.
for i in range(conf_matrix_avg.shape[0]):
    for j in range(conf_matrix_avg.shape[1]):
        plt.text(j, i, f"{int(conf_matrix_avg[i, j])}",
                 horizontalalignment="center",
                 color="white" if conf_matrix_avg[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig(f"ConfusionMatrix_Average_{mapa}_RM_SD.jpeg", format="jpeg")

# Salvar acurácia média
mean_accuracy = np.mean(accuracies)
with open("Accuracies_RM_SD.txt", "a") as f:
    if f.tell() == 0:
        f.write("Accuracies:\n")
    f.write(f"{mapa} = {mean_accuracy:.4f}\n")
