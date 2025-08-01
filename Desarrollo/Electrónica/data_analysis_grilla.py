import os
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

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

# === Primera figura: 8 transiciones (4 filas x 2 columnas) ===
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

# === Segunda figura: 6 transiciones (3 filas x 2 columnas) ===
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
