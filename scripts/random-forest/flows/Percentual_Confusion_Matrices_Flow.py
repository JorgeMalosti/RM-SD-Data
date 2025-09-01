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
conf_matrices_percent = {}
accuracies = []

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
            else:
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
                df = pd.concat([df_rm[['param']], df_rm.drop(columns=['param']), df_sd_final.drop(columns=['param'])], axis=1)

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

        conf_matrix_percent = conf_matrix_total / conf_matrix_total.sum(axis=1, keepdims=True) * 100
        conf_matrices_percent[(metodo, fluxo)] = (conf_matrix_percent, unique_vals)
        accuracy = np.trace(conf_matrix_percent) / 1000  # total 100% por linha, 10 classes
        accuracies.append((metodo, fluxo, accuracy))

fig, axs = plt.subplots(3, 2, figsize=(13, 18))
axs = axs.flatten()

for i, metodo in enumerate(metodos):
    for j, fluxo in enumerate(fluxos):
        idx = i * 2 + j
        conf_matrix_percent, unique_vals = conf_matrices_percent[(metodo, fluxo)]
        ax = axs[idx]
        im = ax.imshow(conf_matrix_percent, interpolation='nearest', cmap=plt.cm.Reds, vmin=0, vmax=100)

        tick_marks = np.arange(len(unique_vals))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels([f"{val:.2f}" for val in unique_vals], rotation=45)
        ax.set_yticklabels([f"{val:.2f}" for val in unique_vals])
        ax.tick_params(axis='both', labelsize=14)

        if j == 0:
            ax.set_ylabel(f"Real", fontsize=18)
        else:
            ax.set_ylabel(f"{linha_labels[i]}", fontsize=18)
            ax.yaxis.set_label_position("right")

        if i == 2:
            ax.set_xlabel("Predicted", fontsize=18)

        if i == 0:
            ax.set_title(titulo_colunas[j], fontsize=18)

        for m in range(conf_matrix_percent.shape[0]):
            for n in range(conf_matrix_percent.shape[1]):
                valor = int(round(conf_matrix_percent[m, n]))
                ax.text(n, m, f"{valor}", ha="center",
                        color="white" if valor > 50 else "black", fontsize=12)

plt.tight_layout(pad=3.0)
fig.subplots_adjust(right=0.88)
pos_top = axs[0].get_position()
pos_bottom = axs[-1].get_position()
cbar_ax = fig.add_axes([0.915, pos_bottom.y0, 0.015, pos_top.y1 - pos_bottom.y0])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.tick_params(labelsize=20)
cbar.set_label("Percentage (%)", fontsize=20, labelpad=0)
plt.savefig("ConfusionMatrices_Lorenz_Rossler.pdf", format="pdf", dpi=300)
