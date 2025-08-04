import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def create_dataframe(path, dataset_names):
    series = []
    for ds_name in dataset_names:
        dir = os.path.join(path, ds_name)
        images = []
        for img in os.listdir(dir):
            if "_" in img:
                if img.startswith("pinza"):
                    img = "pinza"
                elif img.startswith("empty"):
                    img = "empty"
                else:
                    img = img.split("_")[0][:-2]
            else:
                img = img[:-6]
            images.append(img)
        series.append(images)
        # construir DataFrame combinado
    df = pd.DataFrame({
        "label": np.concatenate(series),
        "dataset": [dataset_names[0]] * len(series[0]) + [dataset_names[1]] * len(series[1]) + [dataset_names[2]] * len(series[2])
    })
    return df
    
def plot_distributions(df_data, palette, title, show=True, filename=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    #df = pd.DataFrame({"label": data})

    # gráfico de conteos (frecuencias)
    order = ["empty", "tuerca", "tornillo", "clavo", "lapicera", "tenedor", "cuchara", "destornillador", "martillo", "pinza"]
    sns.countplot(data=df_data, y="label", hue="dataset", order=order, palette=palette)

    # bar_label automáticamente pone el valor en cada barra
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)

    plt.title(title)
    plt.xlabel("Cantidad")
    plt.ylabel("")  # quita el label si no lo quieres
    plt.tight_layout()

    if show:
        plt.show()

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    path = "Desarrollo/simulation/Env04/DataSets"
    output_dir = "Desarrollo/Documentacion/Datasets/"

    dataset_names = ["RawTools_train", "TrainSet", "TrainSet_masks"]
    df = create_dataframe(path, dataset_names)
    plot_distributions(df, palette=("#8f97e4","#45aed8","#0b33b6"), title="Cantidad de imágenes Training Sets", 
                       filename=output_dir + "train_distributions.png")

    dataset_names = ["RawTools_test", "TestSet", "TestSet_masks"]
    df = create_dataframe(path, dataset_names)
    plot_distributions(df, palette=("#d3b7b7","#d88c45","#b60b0b"), title="Cantidad de imágenes Testing Sets",
                       filename=output_dir + "test_distributions.png")