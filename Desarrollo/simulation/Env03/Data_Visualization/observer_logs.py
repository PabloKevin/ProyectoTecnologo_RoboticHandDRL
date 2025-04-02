import polars as pl
import matplotlib.pyplot as plt


# 1. Leer el CSV con polars
log_path = "Desarrollo/simulation/Env03/"
log_name = "observer_logs.csv"
df = pl.read_csv(log_path+log_name)

# 2. Separar las filas de entrenamiento/validación (épocas >= 1)
#    y las filas de test (época = -1)
df_train_val = df.filter(pl.col("epoch") > 0)
df_test      = df.filter(pl.col("epoch") == -1)

df_train = df_train_val.select("run", "epoch", "train_loss")
df_val   = df_train_val.select("run", "epoch", "validation_loss")
df_test  = df_test.select("run", "epoch", "test_loss", "conv_channels", "hidden_layers", "learning_rate")
print(len(df_train))

n_runs = 1
df_train = df_train.filter(pl.col("run")>=df_train["run"].max()-n_runs+1)
df_val   = df_val.filter(pl.col("run")>=df_train["run"].max()-n_runs+1)  
df_test  = df_test.filter(pl.col("run")>=df_train["run"].max()-n_runs+1)

# 5. Graficar en una sola figura
plt.figure()

# - Gráfica de train_loss
plt.plot(range(len(df_train)), df_train["train_loss"], label="Train Loss", color="blue")

# - Gráfica de validation_loss
plt.plot(range(len(df_val)), df_val["validation_loss"], label="Val Loss", color="orange")

# - Gráfica (punto) de test_loss (solo se define en epoch=-1)
#   que forzamos a dibujar en epoch=20
plt.scatter([(i+1)*20-1 for i in range(len(df_test))], df_test["test_loss"], marker="o", label="Test Loss", color="red")

# 6. Personalizar ejes, título y leyenda
plt.xlabel("Epoch")
plt.ylabel("Loss")
conv_channels = df_test["conv_channels"].unique().item()
hidden_layers = df_test["hidden_layers"].unique().item()
learning_rate = df_test["learning_rate"].unique().item()
info_text = f"conv_channels:{conv_channels}\nhidden_layers:{hidden_layers}\nlearning_rate:{learning_rate}"
plt.text(
    0.15, 0.90, 
    info_text,
    transform=plt.gca().transAxes,        # Para usar coords relativas (0..1)
    fontsize=10,
    verticalalignment='top', 
    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
)

plt.title(f"Train / Validation / Test Loss")
plt.legend()

# 7. Mostrar la figura
plt.show()