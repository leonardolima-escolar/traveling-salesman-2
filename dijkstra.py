import numpy as np
import networkx as nx
import heapq
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def dijkstra(graph, start):
    distances = {node: float("infinity") for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight["weight"]
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances


def tsp(graph, start):
    unvisited = set(graph.nodes)
    unvisited.remove(start)
    current_node = start
    tour = [start]

    while unvisited:
        distances = dijkstra(graph, current_node)
        next_node = min(unvisited, key=lambda x: distances[x])
        tour.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node

    tour.append(start)

    return tour


matriz_de_custo = np.genfromtxt("graph2.txt", delimiter=";")

matriz_de_custo[matriz_de_custo == -1] = float("inf")

G = nx.Graph()

for i in range(len(matriz_de_custo)):
    for j in range(len(matriz_de_custo[i])):
        if matriz_de_custo[i, j] != float("inf"):
            G.add_edge(i + 1, j + 1, weight=matriz_de_custo[i, j])

no_inicial = 1
start_time = time.time()
melhor_caminho = tsp(G, no_inicial)
end_time = time.time()
tempo = end_time - start_time

print(f"Tempo: {tempo} segundos")
print("Melhor caminho:", melhor_caminho)


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


custo_atual = calcular_custo(G, melhor_caminho)
print("Custo:", custo_atual)

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


def update(frame):
    if frame < len(melhor_caminho) - 1:
        x = [pos[melhor_caminho[i]][0] for i in range(frame + 2)]
        y = [pos[melhor_caminho[i]][1] for i in range(frame + 2)]
        edges.set_data(x, y)
    return (edges,)


ani = FuncAnimation(
    fig, update, frames=len(melhor_caminho), init_func=init, interval=500, repeat=False
)

plt.show()
