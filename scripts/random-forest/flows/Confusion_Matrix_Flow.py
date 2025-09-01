import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

num_files = 5
metodos = ['RM', 'SD', 'RM+SD']
fluxos = ['Lorenz', 'Rossler']
linha_labels = ['RM', 'SD', 'RM+SD']
titulo_colunas = ['(a) Lorenz', '(b) Rössler']
max_value = 0
conf_matrices_avg = {}
accuracies = []  # Lista para armazenar as acurácias

# Cálculo das matrizes e valor máximo
for metodo in metodos:
    for fluxo in fluxos:
        conf_matrix_total = np.zeros((10, 10), dtype=int)

        for i in range(1, num_files + 1):
            if metodo == 'RM':
                data_rm = np.loadtxt(f"Data_{fluxo}_RM_10classes_{i}.dat")
                df = pd.DataFrame(data_rm, columns=['param'] + [f"Probs_{j}" for j in range(16)])

            elif metodo == 'SD':
                df_raw = pd.read_csv(f"Data_{fluxo}_SD_10classes_{i}.dat", sep="\t")
                df_sd = df_raw[[df_raw.columns[0], df_raw.columns[2]]].copy()
                df_sd.columns = ['param', 'Pattern_Probabilities']
                probs_sd = df_sd["Pattern_Probabilities"].str.split(",", expand=True).astype(float)
                probs_sd.columns = [f"Probs_{j}" for j in range(16)]
                df = pd.concat([df_sd[['param']], probs_sd], axis=1)

            else:  # RM+SD
                data_rm = np.loadtxt(f"Data_{fluxo}_RM_10classes_{i}.dat")
                df_rm = pd.DataFrame(data_rm, columns=['param'] + [f"Probs_RM_{j}" for j in range(16)])

                df_raw = pd.read_csv(f"Data_{fluxo}_SD_10classes_{i}.dat", sep="\t")
                df_sd = df_raw[[df_raw.columns[0], df_raw.columns[2]]].copy()
                df_sd.columns = ['param', 'Pattern_Probabilities']
                probs_sd = df_sd["Pattern_Probabilities"].str.split(",", expand=True).astype(float)
                probs_sd.columns = [f"Probs_SD_{j}" for j in range(16)]
                df_sd_final = pd.concat([df_sd[['param']], probs_sd], axis=1)

                if not np.allclose(df_rm['param'].values, df_sd_final['param'].values):
                    raise ValueError(f"Parâmetros não coincidem para {fluxo}, arquivo {i}")

                df = pd.concat([df_rm[['param']], df_rm.drop(columns=['param']),
                                df_sd_final.drop(columns=['param'])], axis=1)

            unique_vals = np.sort(df['param'].unique())
            param_to_class = {val: idx for idx, val in enumerate(unique_vals)}
            df["Param_class"] = df["param"].map(param_to_class)

            X = df[[col for col in df.columns if "Probs" in col]].copy()
            y = df["Param_class"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            conf_matrix = metrics.confusion_matrix(y_test, y_pred)
            conf_matrix_total += conf_matrix
            max_value = max(max_value, conf_matrix.max())

        conf_matrix_avg = conf_matrix_total / num_files
        conf_matrices_avg[(metodo, fluxo)] = (conf_matrix_avg, unique_vals)

        # Cálculo da acurácia média para essa combinação método/fluxo
        accuracy = np.trace(conf_matrix_avg) / np.sum(conf_matrix_avg)
        accuracies.append((metodo, fluxo, accuracy))

# Salvando as acurácias em arquivo .dat
with open("accuracies_fluxos.dat", "w") as f:
    f.write("Metodo\tFluxo\tAcuracia\n")
    for metodo, fluxo, acc in accuracies:
        f.write(f"{metodo}\t{fluxo}\t{acc:.4f}\n")

# Plot
fig, axs = plt.subplots(3, 2, figsize=(13, 18))
axs = axs.flatten()

for i, metodo in enumerate(metodos):
    for j, fluxo in enumerate(fluxos):
        idx = i * 2 + j
        conf_matrix_avg, unique_vals = conf_matrices_avg[(metodo, fluxo)]
        ax = axs[idx]
        im = ax.imshow(conf_matrix_avg, interpolation='nearest', cmap=plt.cm.Reds, vmin=0, vmax=max_value)

        tick_marks = np.arange(len(unique_vals))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels([f"{val:.2f}" for val in unique_vals], rotation=45)
        ax.set_yticklabels([f"{val:.2f}" for val in unique_vals])
        ax.tick_params(axis='both', labelsize=14)

        # Label de eixo y: "Real" + método
        if j == 0:
            ax.set_ylabel(f"Real", fontsize=18)
        else:
            ax.set_ylabel("")
            ax.set_ylabel(f"{linha_labels[i]}", fontsize=18)
            ax.yaxis.set_label_position("right")

        # Label de eixo x apenas na última linha
        if i == 2:
            ax.set_xlabel("Predicted", fontsize=18)
        else:
            ax.set_xlabel("")

        # Título somente na linha superior
        if i == 0:
            ax.set_title(titulo_colunas[j], fontsize=18)
        else:
            ax.set_title("")

        # Valores dentro da matriz
        thresh = conf_matrix_avg.max() / 2.
        for m in range(conf_matrix_avg.shape[0]):
            for n in range(conf_matrix_avg.shape[1]):
                ax.text(n, m, f"{int(conf_matrix_avg[m, n])}",
                        ha="center",
                        color="white" if conf_matrix_avg[m, n] > thresh else "black", fontsize=12)

# Ajuste layout e obtenha a posição total dos subplots
plt.tight_layout(pad=3.0)
fig.subplots_adjust(right=0.9)  # Reserva espaço para a colorbar

# Obter as coordenadas do primeiro e último eixo (superior e inferior da imagem)
pos_top = axs[0].get_position()
pos_bottom = axs[-1].get_position()

# Coordenadas: [left, bottom, width, height]
cbar_ax = fig.add_axes([
    0.915,                          # left (posição horizontal da barra)
    pos_bottom.y0,                 # bottom (alinhado com o último subplot)
    0.015,                         # width
    pos_top.y1 - pos_bottom.y0    # height (do topo do 1º ao fim do último subplot)
])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.tick_params(labelsize=20)
cbar.set_label("Average number of samples", fontsize=20)

# Salvar figura
plt.savefig("ConfusionMatrices_Lorenz_Rossler.pdf", format="pdf", dpi=300)
