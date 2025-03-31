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

df_train = df_train_val.select("epoch", "train_loss")
df_val   = df_train_val.select("epoch", "validation_loss")
df_test  = df_test.select("epoch", "test_loss")
print(len(df_train))

# 5. Graficar en una sola figura
plt.figure()

# - Gráfica de train_loss
plt.plot(range(len(df_train)), df_train["train_loss"], label="Train Loss", color="blue")

# - Gráfica de validation_loss
plt.plot(range(len(df_val)), df_val["validation_loss"], label="Val Loss", color="orange")

# - Gráfica (punto) de test_loss (solo se define en epoch=-1)
#   que forzamos a dibujar en epoch=20
plt.scatter([19,39], df_test["test_loss"], marker="o", label="Test Loss", color="red")

# 6. Personalizar ejes, título y leyenda
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train / Validation / Test Loss")
plt.legend()

# 7. Mostrar la figura
plt.show()
