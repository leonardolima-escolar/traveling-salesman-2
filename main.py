import itertools
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


matriz_de_custo = np.genfromtxt("graph1.txt", delimiter=";")
# matriz_de_custo = np.genfromtxt("graph2.txt", delimiter=";")
# matriz_de_custo = np.genfromtxt("graph3.txt", delimiter=";")
# matriz_de_custo = np.genfromtxt("graph4.txt", delimiter=";")


G = nx.Graph()

for i in range(len(matriz_de_custo)):
    for j in range(len(matriz_de_custo[i])):
        if matriz_de_custo[i, j] != -1:
            G.add_edge(i + 1, j + 1, weight=matriz_de_custo[i, j])


def calcular_custo(caminho):
    custo_total = 0
    for i in range(len(caminho) - 1):
        if G.has_edge(caminho[i], caminho[i + 1]):
            custo_total += G[caminho[i]][caminho[i + 1]]["weight"]
        else:
            custo_total += float("inf")

    return custo_total


vertices = list(G.nodes())
melhor_caminho = None
melhor_custo = float("inf")

count = 0
start_time = time.time()
for permutacao in itertools.permutations(vertices):
    if permutacao[0] == 1 and G.has_edge(permutacao[-1], permutacao[0]):
        caminho = permutacao + (1,)
        count = count + 1
        if all(G.has_edge(caminho[i], caminho[i + 1]) for i in range(len(caminho) - 1)):
            custo_atual = calcular_custo(caminho)
            if custo_atual < melhor_custo:
                melhor_caminho = caminho
                melhor_custo = custo_atual
print(count)
end_time = time.time()
tempo = end_time - start_time

print(f"Tempo: {tempo} segundos")


print("Melhor caminho:", melhor_caminho)
print("Custo:", melhor_custo)


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

(edges,) = ax.plot([], [], "red", lw=2)


def init():
    edges.set_data([], [])
    return (edges,)


# def update(frame):
#     if frame < len(melhor_caminho) - 1:
#         x = [pos[melhor_caminho[frame]][0], pos[melhor_caminho[frame + 1]][0]]
#         y = [pos[melhor_caminho[frame]][1], pos[melhor_caminho[frame + 1]][1]]
#         edges.set_data(x, y)
#     return (edges,)


def update(frame):
    if frame < len(melhor_caminho) - 1:
        x = [pos[melhor_caminho[i]][0] for i in range(frame + 2)]
        y = [pos[melhor_caminho[i]][1] for i in range(frame + 2)]
        edges.set_data(x, y)
    return (edges,)


ani = FuncAnimation(
    fig, update, frames=len(melhor_caminho), init_func=init, interval=1000, repeat=False
)

plt.show()
