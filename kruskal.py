import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def tsp_kruskal(graph, start):
    edges = list(graph.edges(data=True))
    edges.sort(key=lambda x: x[2]["weight"])

    mst_edges = []
    mst_nodes = set()

    for edge in edges:
        if edge[0] not in mst_nodes or edge[1] not in mst_nodes:
            mst_nodes.add(edge[0])
            mst_nodes.add(edge[1])
            mst_edges.append(edge)

            if len(mst_nodes) == len(graph.nodes):
                break

    cycle = list(nx.dfs_preorder_nodes(nx.Graph(mst_edges), source=start))
    cycle.append(start)

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
melhor_caminho_kruskal = tsp_kruskal(G, no_inicial)
end_time = time.time()
tempo = end_time - start_time

print(f"Tempo: {tempo} segundos")
print("Melhor caminho Kruskal:", melhor_caminho_kruskal)


def calcular_custo(graph, caminho):
    custo_total = 0

    for i in range(len(caminho) - 1):
        if graph.has_edge(caminho[i], caminho[i + 1]):
            custo_total += graph[caminho[i]][caminho[i + 1]]["weight"]
        # else:
        #
        #     custo_total = float("inf")
        #     break

    return custo_total


custo_atual_kruskal = calcular_custo(G, melhor_caminho_kruskal)
print("Custo (Kruskal Adaptado):", custo_atual_kruskal)

fig, ax = plt.subplots()
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_weight="bold", node_size=700, font_size=8)

(edges,) = ax.plot([], [], "red", lw=2)


def init():
    edges.set_data([], [])
    return (edges,)


def update(frame):
    if frame < len(melhor_caminho_kruskal) - 1:
        x = [pos[melhor_caminho_kruskal[i]][0] for i in range(frame + 2)]
        y = [pos[melhor_caminho_kruskal[i]][1] for i in range(frame + 2)]
        edges.set_data(x, y)
    return (edges,)


ani = FuncAnimation(
    fig,
    update,
    frames=len(melhor_caminho_kruskal),
    init_func=init,
    interval=500,
    repeat=False,
)

plt.show()
