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
from networks import CriticNetwork, ActorNetwork

def visualizar_red(modelo: nn.Module):
    G = nx.DiGraph()
    posiciones = {}
    nodo_id = 0
    nodos_por_capa = []

    # Paso 1: determinar cantidad de nodos por capa
    capas_lineales = [m for m in modelo.modules() if isinstance(m, nn.Linear)]
    sizes = []
    for i, capa in enumerate(capas_lineales):
        sizes.append(capa.in_features if i == 0 else capa.out_features)
    sizes.append(capas_lineales[-1].out_features)  # última capa

    for i, cant in enumerate(sizes):
        nodos = []
        # calcular desplazamiento vertical para centrar
        offset = (max(sizes) - cant) / 2
        for j in range(cant):
            G.add_node(nodo_id)
            posiciones[nodo_id] = (i, -j - offset)  # centrado vertical
            nodos.append(nodo_id)
            nodo_id += 1
        nodos_por_capa.append(nodos)

    # Paso 3: crear edges con pesos reales
    max_peso = max(p.weight.abs().max().item() for p in capas_lineales)
    for idx, capa in enumerate(capas_lineales):
        pesos = capa.weight.data
        origen = nodos_por_capa[idx]
        destino = nodos_por_capa[idx + 1]
        for j, n_out in enumerate(destino):
            for i, n_in in enumerate(origen):
                w = pesos[j, i].item()
                color = 'red' if w > 0 else 'blue'
                min_alpha = 0.2  # valor mínimo de opacidad
                alpha = min_alpha + (1 - min_alpha) * (abs(w) / max_peso)
                G.add_edge(n_in, n_out, color=color, alpha=alpha)

    # Paso 4: dibujar
    # Dibujar nodos
    nx.draw_networkx_nodes(G, pos=posiciones, node_size=100, node_color="#66a3c7", edgecolors='black')

    # Dibujar aristas una por una con su color y alpha
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

    # Opcional: agregar título y quitar ejes
    plt.title("Red neuronal con pesos reales (color = signo, opacidad = magnitud)")
    plt.axis('off')
    plt.show()

# 🧠 EJEMPLO DE USO:
# from tu_archivo import CriticNetwork  # o ObserverNetwork
modelo = ActorNetwork(hidden_layers=[64,32,16])
modelo.load_state_dict(torch.load("Desarrollo/simulation/Env04/models_params_weights/td3/actor_td3"))  # si ya lo entrenaste
visualizar_red(modelo)
