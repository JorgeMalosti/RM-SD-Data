import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Lista de mapas e seus parâmetros
mapas = {
    "Betax": "β",
    "Gauss": "γ",
    "Henon": "a",
    "Ikeda": "u",
    "Logistic": "r",
    "Tent": "μ", 
    "Lorenz": "ρ",
    "Rossler": "a"
}

num_files = 5

for mapa, parametro in mapas.items():
    importancias_acumuladas = None  # Vai somar as importâncias dos 5 arquivos

    for i in range(1, num_files + 1):
        file_rm = f"Data_{mapa}_RM_40classes_{i}.dat"
        file_sd = f"Data_{mapa}_SD_40classes_{i}.dat"

        # Carregamento RM
        data_rm = np.loadtxt(file_rm)
        df_rm = pd.DataFrame(data_rm, columns=[parametro] + [f"Probs_RM_{j}" for j in range(16)])

        # Carregamento SD
        df_sd_raw = pd.read_csv(file_sd, sep="\t")
        df_sd = df_sd_raw[[df_sd_raw.columns[0], df_sd_raw.columns[2]]].copy()
        df_sd.columns = [parametro, "Pattern_Probabilities"]
        probs_sd = df_sd["Pattern_Probabilities"].str.split(",", expand=True).astype(float)
        probs_sd.columns = [f"Probs_SD_{j}" for j in range(16)]
        df_sd_final = pd.concat([df_sd[[parametro]], probs_sd], axis=1)

        # Verificação de alinhamento
        if not np.allclose(df_rm[parametro].values, df_sd_final[parametro].values):
            raise ValueError(f"Parâmetros não coincidem entre RM e SD para o mapa {mapa}, arquivo {i}")

        # Combinar dados
        df_combined = pd.concat([
            df_rm[[parametro]],
            df_rm.drop(columns=[parametro]),
            df_sd_final.drop(columns=[parametro])
        ], axis=1)

        # Criar classes baseadas nos parâmetros únicos
        unique_vals = np.sort(df_combined[parametro].unique())
        param_to_class = {val: idx for idx, val in enumerate(unique_vals)}
        df_combined["Param_class"] = df_combined[parametro].map(param_to_class)

        # Separar X e y
        X = df_combined[[col for col in df_combined.columns if "Probs_" in col]].copy()
        y = df_combined["Param_class"]

        # Treinar Random Forest
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        importancias = clf.feature_importances_

        # Acumular importâncias
        if importancias_acumuladas is None:
            importancias_acumuladas = importancias
        else:
            importancias_acumuladas += importancias

    # Média das importâncias ao longo dos arquivos
    importancias_medias = importancias_acumuladas / num_files

    # Salvar em arquivo
    np.savetxt(f"Importance_Features_{mapa}_40classes.dat", importancias_medias)
