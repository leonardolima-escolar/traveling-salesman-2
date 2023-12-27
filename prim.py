import numpy as np
import networkx as nx
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def prim(graph):
    min_spanning_tree = nx.minimum_spanning_tree(graph)
    return min_spanning_tree


def dfs_cycle(tree, start, current_node, visited, cycle):
    visited[current_node] = True
    cycle.append(current_node)

    for neighbor in tree.neighbors(current_node):
        if not visited[neighbor]:
            dfs_cycle(tree, start, neighbor, visited, cycle)

    if current_node == start:
        cycle.append(start)


def transform_tree_to_hamiltonian_cycle(tree, start):
    visited = {node: False for node in tree.nodes}
    cycle = []

    dfs_cycle(tree, start, start, visited, cycle)

    return cycle


matriz_de_custo = np.genfromtxt("graph2.txt", delimiter=";")

matriz_de_custo[matriz_de_custo == -1] = float("inf")


G = nx.Graph()

for i in range(len(matriz_de_custo)):
    for j in range(len(matriz_de_custo[i])):
        if matriz_de_custo[i, j] != float("inf"):
            G.add_edge(i + 1, j + 1, weight=matriz_de_custo[i, j])


no_inicial = 1
start_time = time.time()
min_spanning_tree = prim(G)
melhor_caminho_prim = transform_tree_to_hamiltonian_cycle(min_spanning_tree, no_inicial)
end_time = time.time()
tempo_prim = end_time - start_time

print(f"Tempo (Prim): {tempo_prim} segundos")
print("Melhor caminho (Prim):", melhor_caminho_prim)


def calcular_custo(graph, caminho):
    custo_total = 0

    for i in range(len(caminho) - 1):
        if graph.has_edge(caminho[i], caminho[i + 1]):
            custo_total += graph[caminho[i]][caminho[i + 1]]["weight"]

    return custo_total


custo_atual_prim = calcular_custo(G, melhor_caminho_prim)
print("Custo (Prim):", custo_atual_prim)

fig, ax = plt.subplots()
pos = nx.spring_layout(G)
nx.draw(
    G,
    pos,
    with_labels=True,
    font_weight="bold",
    node_size=700,
    font_size=8,
)

(edges_prim,) = ax.plot([], [], "blue", lw=2)


def init_prim():
    edges_prim.set_data([], [])
    return (edges_prim,)


def update_prim(frame):
    if frame < len(melhor_caminho_prim) - 1:
        x = [pos[melhor_caminho_prim[i]][0] for i in range(frame + 2)]
        y = [pos[melhor_caminho_prim[i]][1] for i in range(frame + 2)]
        edges_prim.set_data(x, y)
    return (edges_prim,)


ani_prim = FuncAnimation(
    fig,
    update_prim,
    frames=len(melhor_caminho_prim),
    init_func=init_prim,
    interval=500,
    repeat=False,
)

plt.show()
