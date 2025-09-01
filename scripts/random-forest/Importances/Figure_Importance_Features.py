import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# -------- CONFIGURÁVEIS ----------
usar_escala_global = False
mapas             = ["Betax", "Gauss", "Henon", "Ikeda", "Logistic", "Tent", "Lorenz", "Rossler"]
titulos           = ["(a) βx", "(b) Gauss", "(c) Hénon", "(d) Ikeda", 
                     "(e) Logistic", "(f) Tent", "(g) Lorenz", "(h) Rössler"]
n_classes_vet     = [10, 20, 40]                 
features_names    = [f'RM{i}' for i in range(1, 17)] + \
                    [f'SD{i}' for i in range(1, 17)]
# -------- /CONFIGURÁVEIS ---------

if usar_escala_global:
    imps = []
    for mapa in mapas:
        for ncls in n_classes_vet:
            arq = f"Importance_Features_{mapa}_{ncls}classes.dat"
            if os.path.exists(arq):
                imps.append(np.loadtxt(arq))
    gmin, gmax = min(i.min() for i in imps), max(i.max() for i in imps)

sns.set(style="whitegrid")

# ---- figura 4x2 ----
fig, axs = plt.subplots(4, 2, figsize=(16, 12), sharex=False)
axs = axs.flatten()

for idx, mapa in enumerate(mapas):
    ax = axs[idx]
    ax.set_title(titulos[idx], fontsize=16, fontweight='bold')

    for j, ncls in enumerate(n_classes_vet):
        y_pos = j + 1
        arq = f"Importance_Features_{mapa}_{ncls}classes.dat"

        if not os.path.exists(arq):
            ax.text(0.5, 0.5, f"{ncls}c\nARQ\nNÃO\nACHADO",
                    ha='center', va='center', fontsize=8, color='red',
                    transform=ax.transAxes)
            continue

        imp = np.loadtxt(arq)
        kw = dict(x=features_names,
                  y=[y_pos]*len(features_names),
                  s=imp * 3500,
                  c=imp,
                  cmap='rainbow',
                  alpha=1,
                  edgecolor='k',
                  linewidth=0.9)
        if usar_escala_global:
            kw.update(vmin=gmin, vmax=gmax)

        sc = ax.scatter(**kw)

    # Ajuste dos rótulos do eixo x para os últimos dois gráficos
    if idx not in (6, 7):
        ax.set_xticklabels([])
        ax.set_xlabel('')
    else:
        # Define ticks e rótulos como números 1-16 e 1-16 para RM e SD
        ax.set_xticks(range(len(features_names)))
        # Rótulos numéricos para os dois grupos separados
        labels = [str(i+1) for i in range(16)]  +  [str(i+1) for i in range(16)]
        ax.set_xticklabels(labels, rotation=0, fontsize=11)
        ax.set_xlabel('')

        # Texto "RM" e "SD" centralizados abaixo do eixo x
        mid_rm = (0 + 15) / 2    # posição central do grupo RM (índices 0 a 15)
        mid_sd = (16 + 31) / 2   # posição central do grupo SD (índices 16 a 31)
        ax.text(mid_rm, -0.20, 'RM', ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax.get_xaxis_transform())
        ax.text(mid_sd, -0.20, 'SD', ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax.get_xaxis_transform())

        # Label geral "Features" abaixo dos dois grupos (mais abaixo que RM e SD)
        ax.text(0.5, -0.35, 'Features', ha='center', va='top', fontsize=14,
                transform=ax.transAxes)

    # Ajustes no eixo y
    if idx % 2 == 0:  # left column
        ax.set_yticklabels([f'{n} classes' for n in n_classes_vet], rotation=45,
                           fontsize=12)
        ax.set_ylabel('')
    else:
        ax.set_yticklabels([])
        ax.set_ylabel('')

    ax.set_xticks(range(len(features_names)))
    ax.set_yticks([1, 2, 3])
    ax.set_ylim(0.5, 3.5)

    # Linhas verticais separando RM (azul) e SD (vermelho)
    mid = len(features_names) // 2
    for i in range(mid):
        ax.axvline(i + 0.5, color='blue', ls='--', lw=0.25)
    for i in range(mid, len(features_names)):
        ax.axvline(i + 0.5, color='red', ls='--', lw=0.25)

# ---------- colorbar global ----------
cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.81])  # left, bottom, width, height
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_label('Importance', fontsize=16)
cbar.ax.tick_params(labelsize=16)  # aumenta o tamanho das fontes dos números da colorbar

plt.subplots_adjust(wspace=0.08, hspace=0.25, right=0.9, bottom=0.18, top=0.93)  # mais bottom para caber texto

plt.savefig("BubblePlot_8Mapas.png", dpi=300, bbox_inches='tight')
