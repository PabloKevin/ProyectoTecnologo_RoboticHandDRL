import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

plt.rcParams["font.size"] = 18

path = "Desarrollo/Documentacion/DocumentsPerRegion/"
df_Asia = pl.read_excel(path + "AsiaticRegion.xlsx")
df_EuropeUnion = pl.read_excel(path + "EuropeUnion.xlsx")
df_EasternEurope = pl.read_excel(path + "EasternEurope.xlsx")
df_LatinAmerica = pl.read_excel(path + "LatinAmerica.xlsx")
df_NorthenAmerica = pl.read_excel(path + "NorthenAmerica.xlsx")

regions = {
    "Asia": df_Asia,
    "Unión Europea": df_EuropeUnion,
    "Europa del Este": df_EasternEurope,
    "Latinoamérica": df_LatinAmerica,
    "Norteamérica": df_NorthenAmerica,
}

labels: list[str] = []
sizes: list[float] = []
explode: list[float] = []

for region, df in regions.items():
    total = float(df["Documents"].sum())
    if region == "Latinoamérica":
        uruguay_total = float(
            df.filter(pl.col("Country") == "Uruguay")["Documents"].sum()
        )
        labels.append(f"{region}\n(Uruguay: {uruguay_total:.0f})")
        explode.append(0.1)
    else:
        labels.append(region)
        explode.append(0.03)
    sizes.append(total)

colors = plt.cm.Pastel1(range(len(sizes)))
fig, ax = plt.subplots(figsize=(12,8))
total_docs = sum(sizes)
autopct = lambda pct: f"{pct * total_docs / 100:.0f}"

wedges, label_texts, value_texts = ax.pie(
    sizes,
    colors=colors,
    labels=labels,
    autopct=autopct,
    startangle=90,
    explode=explode,
    textprops={"fontsize": 18},
    shadow=False,
)

dx, dy = 0.04, -0.04
for wedge in wedges:
    shadow = Wedge(
        center=(wedge.center[0] + dx, wedge.center[1] + dy),
        r=wedge.r,
        theta1=wedge.theta1,
        theta2=wedge.theta2,
        width=wedge.width,
        facecolor="black",
        alpha=0.25,
        zorder=0,
    )
    ax.add_patch(shadow)

for wedge in wedges:
    wedge.set_zorder(1)
    wedge.set_edgecolor("white")

ax.set_title("Cantidad de documentos por región", fontsize=24, fontweight="bold")
ax.axis("equal")

fig.savefig(path + "DocumentsPerRegionPieChart.png", dpi=300, bbox_inches=None)
plt.show()
