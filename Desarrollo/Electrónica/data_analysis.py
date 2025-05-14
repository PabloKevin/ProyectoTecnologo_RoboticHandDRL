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

# Plot current vs time
plt.figure(figsize=(10, 6))
plt.plot(df["time"].to_numpy(), df["current"].to_numpy(), color="blue", label='Current')
#plt.plot([0,600], [0.22, 0.22], color="red", linestyle='--', label='Threshold')
#plt.scatter([t1, t2], [df_peaks["current"][5], df_peaks["current"][6]], color="red", label='Peaks')
plt.scatter(df_peaks["time"].to_numpy(), df_peaks["current"].to_numpy(), color="purple", marker="^", label='Filtered Peaks')
plt.xlabel('Time')
plt.ylabel('Current')
plt.title(f'Current vs Time ({csv_file})')
plt.legend()
plt.grid(True)
plt.show()

# Movimientos individuales
#df_move = df.filter((pl.col("time")>df_peaks["time"][0]-t_min/3) & (pl.col("time")<df_peaks["time"][0]+t_min/3))
window_pre = 1
window_post = 3
df_move = df.filter((pl.col("time")>df_peaks["time"][0]-window_pre) & (pl.col("time")<df_peaks["time"][0]+window_post))

# Plot current vs time
plt.figure(figsize=(10, 6))
plt.scatter(df_move["time"].to_numpy(), df_move["current"].to_numpy(), color="blue", marker=".",label='Current')
plt.plot(df_move["time"].to_numpy(), df_move["current"].to_numpy(), color="blue", label='Current')
#plt.plot([0,600], [0.22, 0.22], color="red", linestyle='--', label='Threshold')
#plt.scatter([t1, t2], [df_peaks["current"][5], df_peaks["current"][6]], color="red", label='Peaks')
plt.scatter(df_peaks["time"][0], df_peaks["current"][0], color="purple", marker="^", label='Filtered Peaks')
plt.xlabel('Time')
plt.ylabel('Current')
plt.title(f'Current vs Time ({csv_file})')
plt.legend()
plt.grid(True)
plt.show()
