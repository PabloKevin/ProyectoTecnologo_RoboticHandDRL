import os
import polars as pl
import matplotlib.pyplot as plt

# Directory containing the CSV files
data_dir = os.path.join(os.path.dirname(__file__), 'Mediciones_Reales')

# List CSV files in the directory
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV files found in Mediciones_Reales directory.")

# Choose a CSV file (for example, the first one)
csv_file = csv_files[4]
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
df2 = df.with_columns(
    (
        (pl.col("current") > 0.2)  # umbral mínimo
        & (pl.col("current") > pl.col("current").shift(1))   # mayor que anterior
        & (pl.col("current") > pl.col("current").shift(-1))  # mayor que siguiente
    ).alias("is_peak")
)

# 2) Contamos los picos
n_peaks = df2.filter(pl.col("is_peak")).count()
print(f"Número de picos > 0.2: {n_peaks}")

# Plot current vs time
plt.figure(figsize=(10, 6))
plt.plot(df["time"].to_numpy(), df["current"].to_numpy(), color="blue", label='Current')
plt.xlabel('Time')
plt.ylabel('Current')
plt.title(f'Current vs Time ({csv_file})')
plt.legend()
plt.grid(True)
plt.show()