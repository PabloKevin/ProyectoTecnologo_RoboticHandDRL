import os
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# Directory containing the CSV files
data_dir = os.path.join(os.path.dirname(__file__), 'Mediciones_Reales')

# List CSV files in the directory
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV files found in Mediciones_Reales directory.")

# Choose a CSV file (for example, the first one)
#csv_file = csv_files[2]
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
        # case A: “0,1,30647252E-09” → parts == ["0","1","30647252E-09"]
        if len(parts) == 3:
            t, c_int, c_frac = parts
            t = float(t)
            c = float(f"{c_int}.{c_frac}")

        # case B: “0,2099866,1,53043994E-09” → parts == ["0","2099866","1","53043994E-09"]
        elif len(parts) == 4:
            t_int, t_frac, c_int, c_frac = parts
            t = float(f"{t_int}.{t_frac}")
            c = float(f"{c_int}.{c_frac}")

        else:
            raise ValueError(f"Unexpected column count ({len(parts)}) in line: {line}")

        times.append(t)
        currents.append(c)

# build a Polars DataFrame
df = pl.DataFrame({
    "time":     times,
    "current":  currents,
})

df = df.filter((pl.col("time")>3) & (pl.col("time")<410))

# 1) Generamos una columna booleana is_peak
df_peaks = df.filter(
        (pl.col("current") > 0.22)  # umbral mínimo
        & (pl.col("current") > pl.col("current").shift(1))   # mayor que anterior
        & (pl.col("current") > pl.col("current").shift(-1))  # mayor que siguiente
)

# tiempo mínimo entre movimientos
t1 = df_peaks["time"][5]
t2 = df_peaks["time"][6]
t_min = t2-t1

indices = []
for i in range(len(df_peaks)-1):
    if df_peaks["time"][i+1] - df_peaks["time"][i] >= t_min:
        if df_peaks["current"][i] > df_peaks["current"][i+1]:
            indices.append(i)
        else:
            indices.append(i+1)

indices = np.unique(indices)
indices = np.append(indices, [0, 9, 13])
indices = np.sort(indices)
#print(f"Indices de picos: {indices}")
df_peaks = df_peaks[indices]

# 2) Contamos los picos
n_peaks = df_peaks.count()
print(f"Número de picos > 0.2: {n_peaks}")

transitions = [ # esto sale de un excel usado durante la práctica para tener un orden de las transiciones
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

# Plot current vs time
plt.figure(figsize=(10, 6))
plt.plot(df["time"].to_numpy(), df["current"].to_numpy(), color="blue", label='Corriente')
#plt.plot([0,600], [0.22, 0.22], color="red", linestyle='--', label='Threshold')
#plt.scatter([t1, t2], [df_peaks["current"][5], df_peaks["current"][6]], color="red", label='Peaks')
plt.scatter(df_peaks["time"].to_numpy(), df_peaks["current"].to_numpy(), color="purple", marker="^", label='Picos de corriente en transiciones')
plt.xlabel('Tiempo (s)')
plt.ylabel('Corriente (A)')
plt.title(f'Corriente en función del tiempo durante transiciones de agarres del prototipo')
plt.legend()
plt.grid(True)

# Anotaciones con flechas para las transiciones
xt = [-20, -30, -22, -10, -22, -20, -20, -20, -15, -20, -50, -45, -20, -20]
yt = [0.06, 0.06, 0.10, 0.06, 0.10, 0.06, 0.10, 0.06, 0.06, 0.06, -0.01, -0.02, 0.06, 0.06]
for i, transition in enumerate(transitions):
    transition = transition.replace(" ", "\n", 1)  # Reemplazar espacios por saltos de línea
    x = df_peaks["time"][i]
    y = df_peaks["current"][i]
    plt.annotate(transition,
                xy=(x, y),         # punto al que apunta
                xytext=(x+xt[i], y+yt[i]),     # posición del texto
                arrowprops=dict(arrowstyle="->", linewidth=1, color="purple"),
                fontsize=10.5, color="purple",)

#plt.show()
plt.savefig(f"Desarrollo/Electrónica/Graficos/All_transitions.png", format="png", dpi=300, bbox_inches="tight")

# Movimientos individuales
#df_move = df.filter((pl.col("time")>df_peaks["time"][0]-t_min/3) & (pl.col("time")<df_peaks["time"][0]+t_min/3))
window_pre = 1
window_post = 3

"""for i, transition in enumerate(transitions):
    df_move = df.filter((pl.col("time")>df_peaks["time"][i]-window_pre) & (pl.col("time")<df_peaks["time"][i]+window_post))

    # Plot current vs time
    plt.figure(figsize=(10, 6))
    plt.scatter(df_move["time"].to_numpy(), df_move["current"].to_numpy(), color="blue", marker=".",label='Current')
    plt.plot(df_move["time"].to_numpy(), df_move["current"].to_numpy(), color="blue", label='Current')
    #plt.plot([0,600], [0.22, 0.22], color="red", linestyle='--', label='Threshold')
    #plt.scatter([t1, t2], [df_peaks["current"][5], df_peaks["current"][6]], color="red", label='Peaks')
    plt.annotate(f'Imax={df_peaks["current"][i]*1000:.2f} mA', 
                 xy=(df_peaks["time"][i], df_peaks["current"][i]),
                 xytext=(df_peaks["time"][i]+0.5, df_peaks["current"][i]), 
                 color="purple",
                 arrowprops=dict(arrowstyle="->", linewidth=1, color="purple"),
                 fontsize=10.5)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Corriente (A)')
    plt.title(f'Pico de corriente en transición: {transition}')
    plt.legend()
    plt.grid(True)
    #plt.show()
    
    #plt.savefig(f"Desarrollo/Electrónica/Graficos/{transition.replace(' ', '_')}.png", format="png", dpi=300, bbox_inches="tight")"""

window_pre = 1
window_post = 5
i=12
df_move = df.filter((pl.col("time")>df_peaks["time"][i]-window_pre) & (pl.col("time")<df_peaks["time"][i]+window_post))

# Plot current vs time
plt.figure(figsize=(10, 6))
plt.scatter(df_move["time"].to_numpy(), df_move["current"].to_numpy(), color="blue", marker=".",label='Current')
plt.plot(df_move["time"].to_numpy(), df_move["current"].to_numpy(), color="blue", label='Current')
#plt.plot([0,600], [0.22, 0.22], color="red", linestyle='--', label='Threshold')
#plt.scatter([t1, t2], [df_peaks["current"][5], df_peaks["current"][6]], color="red", label='Peaks')
plt.annotate(f'Imax={df_peaks["current"][i]*1000:.2f} mA', 
                xy=(df_peaks["time"][i], df_peaks["current"][i]),
                xytext=(df_peaks["time"][i]+0.5, df_peaks["current"][i]), 
                color="purple",
                arrowprops=dict(arrowstyle="->", linewidth=1, color="purple"),
                fontsize=10.5)
plt.xlabel('Tiempo (s)')
plt.ylabel('Corriente (A)')
plt.title(f'Pico de corriente en transición: {transitions[12]}')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig("Desarrollo/Electrónica/Graficos/empty_a_completo.png", format="png", dpi=300, bbox_inches="tight")