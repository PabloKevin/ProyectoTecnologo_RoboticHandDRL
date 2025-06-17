import sys, os
# Ruta del archivo actual
current_dir = os.path.dirname(__file__)
# Un nivel arriba (donde está networks.py)
parent_dir = os.path.join(current_dir, "..")
sys.path.append(os.path.abspath(parent_dir))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from networks import CriticNetwork, ActorNetwork

def visualizar_red(modelo: nn.Module):
    G = nx.DiGraph()
    posiciones = {}
    nodo_id = 0
    nodos_por_capa = []

    # Paso 1: identificar capas lineales
    capas_lineales = [m for m in modelo.modules() if isinstance(m, nn.Linear)]

    # Paso 2: cantidad de nodos por capa (sin nodo de salida adicional)
    sizes = [capas_lineales[0].in_features] + [c.out_features for c in capas_lineales]

    # Paso 3: crear nodos y posiciones centradas
    for i, cant in enumerate(sizes):
        nodos = []
        offset = (max(sizes) - cant) / 2
        for j in range(cant):
            G.add_node(nodo_id)
            posiciones[nodo_id] = (i, -j - offset)
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
    max_salida_total = max(
        capa.weight.data.sum(dim=0).abs().max().item()
        for capa in capas_lineales
    )

    for idx, capa in enumerate(capas_lineales):
        pesos = capa.weight.data
        for i, node_id in enumerate(nodos_por_capa[idx]):
            suma = pesos[:, i].sum().item()  # salientes desde esta neurona
            alpha = min_alpha + (1 - min_alpha) * abs(suma) / max_salida_total
            color_map = cm.Reds if suma > 0 else cm.Blues
            rgba = color_map(alpha)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
            )
            node_colors[node_id] = hex_color

    # Última capa (sin salidas): color neutro
    for node_id in nodos_por_capa[-1]:
        node_colors[node_id] = "#cccccc"

    # Paso 6: dibujar nodos y conexiones
    nx.draw_networkx_nodes(
        G,
        pos=posiciones,
        node_size=100,
        node_color=[node_colors.get(n, "#cccccc") for n in G.nodes()],
        edgecolors='black'
    )

    for u, v in G.edges():
        color = G[u][v]['color']
        alpha = G[u][v]['alpha']
        nx.draw_networkx_edges(
            G,
            pos=posiciones,
            edgelist=[(u, v)],
            edge_color=color,
            alpha=alpha,
            width=1
        )

    plt.title("Red neuronal con pesos reales\nColor de nodo: suma de pesos salientes | Aristas: peso (signo y magnitud)")
    plt.axis('off')
    plt.show()

# 🧠 USO
modelo = ActorNetwork(hidden_layers=[64, 32, 16])
modelo.load_state_dict(torch.load("Desarrollo/simulation/Env04/models_params_weights/td3/actor_td3"))
modelo.eval()
visualizar_red(modelo)
