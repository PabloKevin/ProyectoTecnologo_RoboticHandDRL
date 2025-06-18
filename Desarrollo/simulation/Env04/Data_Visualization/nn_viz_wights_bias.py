import sys, os
# Ruta del archivo actual
current_dir = os.path.dirname(__file__)
# Un nivel arriba (donde está networks.py)
parent_dir = os.path.join(current_dir, "..")
sys.path.append(os.path.abspath(parent_dir))
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from networks import CriticNetwork, ActorNetwork

def anotar_inputs(ax, posiciones, input_dims, x=0, ancho=2.0, curvatura=3.0):
    """
    Dibuja una llave tipo { más curvada, desde y1 a y2 en la posición x.
    - `ancho` es cuánto sobresale horizontalmente
    - `curvatura` determina lo "curvada" que se ve
    """
    y1 = posiciones[0][1]
    y2 = posiciones[9][1]
    mid = (y1 + y2) / 2
    dis = abs(y2 - y1)

    verts = [
        (x + ancho, y2),            # inicio arriba derecha
        (x, y2+dis*0.1),            # inicio arriba derecha
        (x-curvatura*0.1, mid),        # curva arriba
        (x, y1-dis*0.1),            # inicio arriba derecha
        (x + ancho, y1)             # final abajo derecha
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3
    ]

    path = Path(verts, codes)
    patch = PathPatch(path, lw=2, edgecolor='black', facecolor='none')
    ax.add_patch(patch)

    h = [mid, posiciones[10][1], posiciones[11][1]] if input_dims==12 else [mid, posiciones[10][1]]
    for j in h:
        ax.annotate(
            '',                      # sin texto
            xy=(x-curvatura*1.3, j),       # punta de la flecha
            xytext=(x+curvatura*0.7, j),   # base de la flecha
            fontsize=12,
            arrowprops=dict(
                arrowstyle='->',
                color='black',
                lw=2
            )
        )

    ax.annotate(
            '',                      # sin texto
            xy=(posiciones[-1][0]+8, posiciones[-1][1]),       # punta de la flecha
            xytext=(posiciones[-1][0]+2, posiciones[-1][1]),   # base de la flecha
            fontsize=12,
            arrowprops=dict(
                arrowstyle='->',
                color='black',
                lw=2
            )
        )

    plt.text(
        -14,
        mid+2.0,  # Espaciado vertical hacia abajo
        "logits",
        ha='center',
        va='top',
        fontsize=12
    )

    plt.text(
        -19,
        posiciones[10][1]+2.5,  # Espaciado vertical hacia abajo
        "finger index",
        ha='center',
        va='top',
        fontsize=12
    )

    if input_dims==12: 
        plt.text(
            -13,
            posiciones[11][1]+1.5,  # Espaciado vertical hacia abajo
            "state",
            ha='center',
            va='top',
            fontsize=12
        )
        plt.text(
            posiciones[-1][0]+14.5,
            posiciones[-1][1]+1.8,  # Espaciado vertical hacia abajo
            "Q-value",
            ha='center',
            va='top',
            fontsize=12
        )
    elif input_dims==11:
        plt.text(
            posiciones[-1][0]+18,
            posiciones[-1][1]+1.8,  # Espaciado vertical hacia abajo
            "finger action",
            ha='center',
            va='top',
            fontsize=12
        )




def visualizar_red(modelo: nn.Module, name, act_func):
    G = nx.DiGraph()
    posiciones = {}
    nodo_id = 0
    nodos_por_capa = []

    # Paso 1: identificar capas lineales
    capas_lineales = [m for m in modelo.modules() if isinstance(m, nn.Linear)]

    # Paso 2: cantidad de nodos por capa (sin nodo de salida adicional)
    sizes = [capas_lineales[0].in_features] + [c.out_features for c in capas_lineales]

    # Paso 3: crear nodos y posiciones centradas
    espaciado_vertical = 2.0
    espaciado_horizontal = 45.0

    max_nodes = max(sizes)

    posiciones = {}
    nodo_id = 0
    nodos_por_capa = []

    for i, cant in enumerate(sizes):
        nodos = []
        for j in range(cant):
            x = i * espaciado_horizontal
            y = - (j - (cant - 1) / 2) * espaciado_vertical
            posiciones[nodo_id] = (x, y)
            G.add_node(nodo_id)
            nodos.append(nodo_id)
            nodo_id += 1
        nodos_por_capa.append(nodos)


    # Paso 4: crear aristas con color y alpha según pesos
    min_alpha = 0.1
    max_peso = max(p.weight.abs().max().item() for p in capas_lineales)

    for idx, capa in enumerate(capas_lineales):
        pesos = capa.weight.data
        origen = nodos_por_capa[idx]
        destino = nodos_por_capa[idx + 1]
        for j, n_out in enumerate(destino):
            for i, n_in in enumerate(origen):
                w = pesos[j, i].item()
                color = 'red' if w > 0 else 'blue'
                alpha = min_alpha + (1 - min_alpha) * (abs(w) / max_peso)
                G.add_edge(n_in, n_out, color=color, alpha=alpha)

    # Paso 5: calcular colores de nodos por suma de pesos salientes
    node_colors = {}
    min_alpha = 0.2

    # 1. Capa de entrada: color por suma de pesos salientes
    pesos_entrada = capas_lineales[0].weight.data
    max_salida_entrada = pesos_entrada.sum(dim=0).abs().max().item()

    for i, node_id in enumerate(nodos_por_capa[0]):
        suma = pesos_entrada[:, i].sum().item()
        alpha = min_alpha + (1 - min_alpha) * abs(suma) / max_salida_entrada
        color_map = cm.Reds if suma > 0 else cm.Blues
        rgba = color_map(alpha)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
        )
        node_colors[node_id] = hex_color

    # 2. Resto de las capas: color por bias
    # Recorremos capas ocultas + salida
    max_bias = max(c.bias.abs().max().item() for c in capas_lineales)

    for idx, capa in enumerate(capas_lineales):
        bias = capa.bias.data
        destino = nodos_por_capa[idx + 1]
        for i, node_id in enumerate(destino):
            b = bias[i].item()
            alpha = min_alpha + (1 - min_alpha) * abs(b) / max_bias
            color_map = cm.Greens if b > 0 else cm.Greys
            rgba = color_map(alpha)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
            )
            node_colors[node_id] = hex_color


    # Paso 6: dibujar nodos y conexiones
    nx.draw_networkx_nodes(
        G,
        pos=posiciones,
        node_size=100,
        node_color=[node_colors.get(n, "#cccccc") for n in G.nodes()],
        edgecolors='black'
    )

    # Dibujar conexiones con curvas Bézier
    ax = plt.gca()
    for u, v in G.edges():
        x1, y1 = posiciones[u]
        x2, y2 = posiciones[v]
        color = G[u][v]['color']
        alpha = G[u][v]['alpha']

        # Calcular puntos de control para una curva Bézier de grado 3 (curva en "S")
        dx = (x2 - x1) * 0.4  # ajuste de curvatura horizontal
        control1 = (x1 + dx, y1)
        control2 = (x2 - dx, y2)

        # Crear el Path de la curva Bézier
        vertices = [ (x1, y1), control1, control2, (x2, y2) ]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(vertices, codes)

        # Dibujar la curva
        patch = PathPatch(
            path,
            facecolor='none',
            edgecolor=color,
            lw=0.5,
            alpha=alpha
        )
        ax.add_patch(patch)

    # Calcular límites manuales
    x_max = espaciado_horizontal * (len(sizes) - 1) + 10
    y_max = (max_nodes - 1) / 2 * espaciado_vertical +10

    plt.title(f"Visualización de la arquitectura y pesos de la red neuronal {name}", fontsize=16)

    # Aristas: peso (color)
    red_line = Line2D([0], [0], color='red', lw=2, label='Peso positivo')
    blue_line = Line2D([0], [0], color='blue', lw=2, label='Peso negativo')

    # Nodos: suma de pesos salientes
    node_red = Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=cm.Reds(0.8),
                  markeredgecolor='black',
                  markersize=10,
                  label='Nodo: suma pesos salientes es positiva')
    node_blue = Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cm.Blues(0.8),
                   markeredgecolor='black',
                   markersize=10,
                   label='Nodo: suma pesos salientes es negativa')
    node_green = Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cm.Greens(0.8),
                   markeredgecolor='black',
                   markersize=10,
                   label='Nodo: bias positivos')
    node_grey = Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cm.Greys(0.8),
                   markeredgecolor='black',
                   markersize=10,
                   label='Nodo: bias negativos')

    # Texto explicativo sin símbolo
    opacity_note = Line2D([0], [0], color='none', label='Opacidad ∝ magnitud')

    # Mostrar la leyenda
    plt.legend(
        handles=[red_line, blue_line, node_red, node_blue, node_green, node_grey, opacity_note],
        loc='upper right',            # posición base de referencia
        bbox_to_anchor=(1, 0.98), # coordenadas X e Y relativas al gráfico
        frameon=True,
        fontsize=12,
    )

    y_layers = [20,3,20,30,40]
    for i, n in enumerate(sizes):
        if i==0:
            t = f"Input Layer ∈ ℝ^{n}"
        elif i == len(sizes) - 1:
            try:
                t = f"Output Layer ∈ ℝ^{n} + {act_func[i-1]}"
            except:
                t = f"Output Layer ∈ ℝ^{n}"
        else:
            t = f"Hidden Layer ∈ ℝ^{n} + {act_func[i-1]}"
        plt.text(
            x_max/4*i,
            -y_max+y_layers[i],  # Espaciado vertical hacia abajo
            t,
            ha='center',
            va='top',
            fontsize=12
        )

    anotar_inputs(ax, list(posiciones.values()), input_dims=sizes[0], x=-5)

    plt.xlim(-10, x_max)
    plt.ylim(-y_max, y_max)
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')  # asegura relación 1:1 real
    #plt.tight_layout()
    plt.show()



# USO
modelo = ActorNetwork(hidden_layers=[64, 32, 16])
#modelo.load_state_dict(torch.load("Desarrollo/simulation/Env04/models_params_weights/td3/target_actor_td3"))

#modelo = CriticNetwork(input_dims=12, hidden_layers=[64, 32, 16])
modelo.load_state_dict(torch.load("Desarrollo/simulation/Env04/models_params_weights/td3/actor_td3"))

modelo.eval()
#visualizar_red(modelo, name="Target Actor TD3", act_func=["leaky_ReLU", "leaky_ReLU", "leaky_ReLU", "tanh"])
visualizar_red(modelo, name="Actor TD3", act_func=["leaky_ReLU", "leaky_ReLU", "leaky_ReLU"])
