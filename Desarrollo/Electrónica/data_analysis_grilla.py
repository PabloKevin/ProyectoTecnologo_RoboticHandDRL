import os
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

# === Configuración de rutas ===
data_dir = os.path.join(os.path.dirname(__file__), 'Mediciones_Reales')
output_dir = os.path.join("Desarrollo", "Electrónica", "Graficos")
os.makedirs(output_dir, exist_ok=True)

# ==== Leer CSV ====
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV files found in Mediciones_Reales directory.")

# Puedes cambiar por otro archivo si quieres
csv_file = "2025-04-01_162510343_DC Current(A)pk2.csv"
csv_path = os.path.join(data_dir, csv_file)
print(f"Reading {csv_path}...")

times = []
currents = []

with open(csv_path, "r") as f:
    for raw in f:
        line = raw.strip()
        if not line:
            continue

        parts = line.split(",")
        if len(parts) == 3:
            t, c_int, c_frac = parts
            t = float(t)
            c = float(f"{c_int}.{c_frac}")
        elif len(parts) == 4:
            t_int, t_frac, c_int, c_frac = parts
            t = float(f"{t_int}.{t_frac}")
            c = float(f"{c_int}.{c_frac}")
        else:
            raise ValueError(f"Unexpected column count ({len(parts)}) in line: {line}")

        times.append(t)
        currents.append(c)

# Construir DataFrame Polars
df = pl.DataFrame({
    "time":     times,
    "current":  currents,
})

# Filtrar rango de interés
df = df.filter((pl.col("time") > 3) & (pl.col("time") < 410))

# 1) Detectar picos
df_peaks = df.filter(
    (pl.col("current") > 0.22)
    & (pl.col("current") > pl.col("current").shift(1))
    & (pl.col("current") > pl.col("current").shift(-1))
)

# tiempo mínimo entre movimientos: usar los picos 5 y 6 como en tu lógica original
if df_peaks.height < 7:
    raise RuntimeError("No hay suficientes picos detectados para aplicar la lógica de filtrado.")

t1 = df_peaks["time"][5]
t2 = df_peaks["time"][6]
t_min = t2 - t1

indices = []
for i in range(len(df_peaks) - 1):
    if df_peaks["time"][i+1] - df_peaks["time"][i] >= t_min:
        if df_peaks["current"][i] > df_peaks["current"][i+1]:
            indices.append(i)
        else:
            indices.append(i+1)

indices = np.unique(indices)
# agregar índices fijos como tenías
indices = np.append(indices, [0, 9, 13])
indices = np.sort(indices)
df_peaks = df_peaks[indices]

# 2) Contar picos
n_peaks = df_peaks.height
print(f"Número de picos > 0.2 (después de filtrado): {n_peaks}")

# Lista de transiciones (de tu Excel)
transitions = [
    "empty a martillo",
    "martillo a lapicera",
    "lapicera a martillo",
    "martillo a tornillo",
    "tornillo a martillo",
    "martillo a empty",
    "empty a lapicera",
    "lapicera a tornillo",
    "tornillo a lapicera",
    "lapicera a empty",
    "empty a tornillo",
    "tornillo a empty",
    "empty a completo",
    "completo a empty",
]

# === Función auxiliar para graficar una transición ===
def plot_single_transition(ax, df, df_peaks, idx, transition, window_pre=1, window_post=3):
    t_peak = df_peaks["time"][idx]
    current_peak = df_peaks["current"][idx]
    df_move = df.filter(
        (pl.col("time") > t_peak - window_pre) &
        (pl.col("time") < t_peak + window_post)
    )

    ax.plot(df_move["time"].to_numpy(), df_move["current"].to_numpy(),
            label='Corriente', marker='.', linewidth=1, color='blue')
    ax.annotate(f'Imax={current_peak*1000:.2f} mA',
                xy=(t_peak, current_peak),
                xytext=(t_peak + 0.5, current_peak),
                arrowprops=dict(arrowstyle="->", linewidth=1, color="purple"),
                fontsize=9, color="purple")
    ax.set_title(transition, fontsize=11)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Corriente (A)")
    ax.grid(True)

"""
# === Figuras de 4 transiciones (2 filas x 2 columnas) ===
for i in range(0,3):
    first_block = transitions[0+i*4:4+i*4]
    print(first_block)
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes1_flat = axes1.flatten()
    for j, transition in enumerate(first_block):
        if j >= df_peaks.height:
            break
        plot_single_transition(axes1_flat[j], df, df_peaks, j+4*i, transition)
    fig1.suptitle("Picos de corriente en transiciones", fontsize=12)
    out1 = os.path.join(output_dir, f"combined_transitions_{i}.png")
    fig1.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Guardado: {out1}")

# === Figura de 2 transiciones (2 filas x 1 columnas) ===
second_block = transitions[12:14]
offset = 12
fig2, axes2 = plt.subplots(2, 1, figsize=(9, 12), constrained_layout=True)
axes2_flat = axes2.flatten()
for j, transition in enumerate(second_block):
    idx = offset + j
    if idx >= df_peaks.height:
        break
    plot_single_transition(axes2_flat[j], df, df_peaks, idx, transition, window_post=5)
# ocultar ejes sobrantes si los hubiera
for k in range(len(second_block), len(axes2_flat)):
    axes2_flat[k].set_visible(False)
fig2.suptitle("Picos de corriente en transiciones", fontsize=12)
out2 = os.path.join(output_dir, "combined_transitions_3.png")
fig2.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Guardado: {out2}")
"""

# BOXPLOT DE PICOS DE CORRIENTE
# Asume que df_peaks ya está definido y tiene la columna "current"

def boxplot(data, color, text_color, title, figsize=(6,6), filename=None, hist=False):
    # Posiciones: puntos a la izquierda (x≈0.8), boxplot a la derecha (x=1.1)
    x_box = 1.1
    x_points_center = 0.8

    fig, ax = plt.subplots(figsize=figsize)

    # 1. Boxplot (a la derecha)
    bp = ax.boxplot(
        data,
        positions=[x_box],
        widths=0.3,
        patch_artist=True,
        showfliers=False,  # ocultamos outliers porque ya los mostramos como puntos
        boxprops=dict(facecolor=color, alpha=0.7, edgecolor="none"),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="gray", linewidth=1),
        capprops=dict(color="gray", linewidth=1),
    )

    # 2. Puntos individuales (a la izquierda) con jitter horizontal
    rng = np.random.default_rng(seed=46)
    x_jitter = rng.normal(loc=x_points_center, scale=0.03, size=len(data))
    ax.scatter(x_jitter, data,
            color=color, alpha=0.8, s=40,
            edgecolors="none", label="Picos individuales")

    # 3. Línea punteada de la media (dentro del boxplot)
    mean_val = data.mean()
    q1 = np.percentile(data, 25, method="linear") 
    q3 = np.percentile(data, 75, method="linear")

    ax.hlines(mean_val, x_box - 0.15, x_box + 0.15,
            colors=text_color, linestyles="--", linewidth=2, label="Media")

    # Etiquetas y formato
    ax.set_xticks([])              # no mostrar ticks duplicados
    ax.set_ylabel("Corriente (A)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.7)

    # Leyenda
    #ax.legend(loc="upper right", frameon=False)

    # Ajuste de límites para que se vean separados y no cortados
    x_min = x_points_center - 0.2
    x_max = x_box + 0.4
    ax.set_xlim(x_min, x_max)

    # Opcional: mostrar valor de la media
    ax.text(x_box + 0.16, mean_val, f"mean={mean_val:.3f}", va="center", fontsize=10, color=text_color)
    ax.text(x_box + 0.16, np.median(data), f"median={np.median(data):.3f}", va="center", fontsize=10, color=text_color)
    ax.text(x_box + 0.16, data.min(), f"min={data.min():.3f}", va="center", fontsize=10, color=text_color)
    ax.text(x_box + 0.16, data.max(), f"max={data.max():.3f}", va="center", fontsize=10, color=text_color)
    ax.text(x_box + 0.16, q1, f"Quartile_1={q1:.3f}", va="center", fontsize=10, color=color)
    ax.text(x_box + 0.16, q3, f"Quartile_3={q3:.3f}", va="center", fontsize=10, color=color)

    if hist:    
        divider = make_axes_locatable(ax)
        ax_hist = divider.append_axes("right", size="25%", pad=0.1, sharey=ax)

        # Histograma horizontal de la misma variable
        sns.histplot(
        data=pl.DataFrame({"current": data}),
        y="current",
        bins=70,
        kde=True,                     # equivalente a distplot: hist + KDE
        stat="count",
        ax=ax_hist,
        color=color,
        alpha=0.7,
        line_kws={"linewidth": 1},
        )

        ax_hist.tick_params(axis="y", which="both", labelleft=False, left=False)
        ax_hist.set_ylabel("")              # quitar cualquier label redundante


    plt.tight_layout()
    plt.show()

    if filename is not None:
        out3 = os.path.join(output_dir, filename)
        fig.savefig(out3, dpi=300, bbox_inches="tight")

peaks = df_peaks["current"].to_numpy()
boxplot(peaks, color="#4db6ac", text_color="darkgreen", title="Distribución de picos de corriente", filename="boxplot_Ipeaks.png")

current = df.filter(pl.col("current") < 0.16)
current = current["current"].to_numpy()
boxplot(current, color="#4d58b6", text_color="darkblue", title="Distribución de Corriente < 0.16 A", 
        filename="boxplot_I.png", figsize=(7, 6), hist=True)
