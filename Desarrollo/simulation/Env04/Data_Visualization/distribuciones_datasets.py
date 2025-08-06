import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from matplotlib.patches import Wedge

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


def calculate_percentages(df_data, pies=[("RawTools",0), ("Set",-1), ("masks",-1)]):
    datasets = df_full["dataset"].unique()
    percentages = []
    counts = []
    for pie, dir in pies:
        ds_names = []
        for ds in datasets:
            if dir == 0:
                if ds.startswith(pie):
                    ds_names.append(ds)
            elif dir == -1:
                if ds.endswith(pie):
                    ds_names.append(ds)
        aux_df = []
        for ds in ds_names:
            aux_df.append(df_data[df_data["dataset"] == ds].copy())
        counts_df = []
        for df in aux_df:
            counts_df.append(df["label"].value_counts())
        counts_df = pd.concat(counts_df)

        per = counts_df / counts_df.sum() * 100
        percentages.append(per)
        counts.append(counts_df.sum())
    return percentages, counts

def make_autopct(df_size):
    def my_autopct(pct):
        counts = int(round(pct * df_size / 100))
        return f"{counts}\n{pct:.1f}%"
    return my_autopct

def pieplot(data, counts, title, show=True, filename=None, palette=("Blues", "Reds")):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    labels = data.index
    sizes = data.values
    n = int(len(labels)/2)
    explode = ([0.06] * n + [0.06] * n)

    for i,lab in enumerate(labels):
        if lab=="empty":
            explode[i] = 0.2

    if palette[1]=="yellow":
        pal = sns.color_palette(palette[0], n) + sns.light_palette(palette[1], n_colors=n) 
        facecolor2 = "#eff315"
    else:
        pal = sns.color_palette(palette[0], n) + sns.color_palette(palette[1], n)
        facecolor2 = sns.color_palette(palette[1], n_colors=1)[0]

    # Graficar el pie chart
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, explode=explode, autopct=make_autopct(counts),
            shadow=False, startangle=90, colors=pal, pctdistance=0.9)
    
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # 1) Parámetro de desplazamiento de la sombra en unidades de data
    dx, dy = -0.02, -0.01

    # 2) Dibujar todas las sombras primero (zorder=0)
    for w in wedges:
        shadow = Wedge(
            center=(w.center[0] + dx, w.center[1] + dy),
            r=w.r,
            theta1=w.theta1,
            theta2=w.theta2,
            width=w.width,
            facecolor="black",
            alpha=0.3,
            zorder=0,
        )
        ax.add_patch(shadow)

    # 3) Elevar los wedges reales por encima de las sombras (zorder=1)
    for w in wedges:
        w.set_zorder(1)

    # Creamos dos parches para la leyenda
    per = np.sum(sizes[:sizes.size//2])

    legend_handles = [
        Patch(facecolor=sns.color_palette(palette[0], n_colors=1)[0], edgecolor="gray", 
              label=f"TrainSet, images={round(per*counts/100)} \n       size={per:.1f}%"),
        Patch(facecolor=facecolor2,  edgecolor="gray", 
              label=f"TestSet, images={round((100-per)*counts/100)} \n      size={100-per:.1f}%")
    ]

    # Añadimos la leyenda al gráfico
    ax.legend(
        handles=legend_handles,
        title="Dataset",
        loc="upper right",
        bbox_to_anchor=(1.05, 1.14),
        frameon=True,
        framealpha=0.9
    )

    plt.title(f"{title} Dataset Distribution", pad=20)
    plt.tight_layout()

    if show:
        plt.show()

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight",transparent=True)


if __name__ == "__main__":
    path = "Desarrollo/simulation/Env04/DataSets"
    output_dir = "Desarrollo/Documentacion/Datasets/"

    dataset_names = ["RawTools_train", "TrainSet", "TrainSet_masks"]
    df_train = create_dataframe(path, dataset_names)
    #plot_distributions(df_train, palette=("#8f97e4","#45aed8","#0b33b6"), title="Cantidad de imágenes Training Sets", 
    #                   filename=output_dir + "train_distributions.png")

    dataset_names = ["RawTools_test", "TestSet", "TestSet_masks"]
    df_test = create_dataframe(path, dataset_names)
    #plot_distributions(df_test, palette=("#d3b7b7","#d88c45","#b60b0b"), title="Cantidad de imágenes Testing Sets",
    #                   filename=output_dir + "test_distributions.png")

    df_full = pd.concat([df_train, df_test], ignore_index=True)
    #print(calculate_percentages(df_full))
    percentages, counts = calculate_percentages(df_full)
    titles = ["RawTools", "Augmented", "Masked-Augmented"]

    for i, per in enumerate(percentages):
        filename = output_dir + f"pieplot_{titles[i]}.png"
        palette = [("Blues", "Reds"), ("Purples", "Oranges"), ("Greens", "yellow")]
        pieplot(per, title=titles[i], filename=filename, palette=palette[i], counts = counts[i])
    