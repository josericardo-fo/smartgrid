import threading
import time
from collections import defaultdict, deque

import dash
import networkx as nx
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output


class ChargingGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.capacity = {}

    def add_edge(self, u, v, cap):
        self.graph[u].append(v)
        self.graph[v].append(u)
        self.capacity[(u, v)] = cap
        self.capacity[(v, u)] = 0

    def bfs(self, source, sink, parent):
        visited = set()
        queue = deque([source])
        visited.add(source)

        while queue:
            u = queue.popleft()
            for v in self.graph[u]:
                if v not in visited and self.capacity[(u, v)] > 0:
                    queue.append(v)
                    visited.add(v)
                    parent[v] = u
                    if v == sink:
                        return True
        return False

    def edmonds_karp(self, source, sink):
        parent = {}
        max_flow = 0

        while self.bfs(source, sink, parent):
            path_flow = float("Inf")
            s = sink

            while s != source:
                path_flow = min(path_flow, self.capacity[(parent[s], s)])
                s = parent[s]

            v = sink
            while v != source:
                u = parent[v]
                self.capacity[(u, v)] -= path_flow
                self.capacity[(v, u)] += path_flow
                v = parent[v]

            max_flow += path_flow

        return max_flow

    def create_networkx_graph(self):
        G = nx.DiGraph()
        for u in self.graph:
            for v in self.graph[u]:
                if (u, v) in self.capacity and self.capacity[(u, v)] > 0:
                    G.add_edge(u, v, capacity=self.capacity[(u, v)])
        return G


# Informações de bateria dos carros em porcentagem
car_batteries = {
    "Car1": 50,  # Carro 1 com 50% de bateria
    "Car2": 30,  # Carro 2 com 30% de bateria
    "Car3": 85,  # Carro 3 com 85% de bateria
    "Car4": 20,  # Carro 4 com 20% de bateria
    "Car5": 95,  # Carro 5 com 45% de bateria
    "Car6": 60,  # Carro 6 com 60% de bateria
    "Car7": 25,  # Carro 7 com 25% de bateria
    "Car8": 75,  # Carro 8 com 70% de bateria
}


# Função para definir a capacidade baseada na carga da bateria
def get_capacity(battery_percentage):
    if battery_percentage < 80:
        return 7.4  # Capacidade alta para baterias < 80% (em kW)
    else:
        return 2.0  # Capacidade baixa para baterias >= 80% (em kW)


# Função para simular o carregamento das baterias
def charge_batteries():
    while True:
        time.sleep(10)  # Espera 10 segundos
        for car in car_batteries:
            if car_batteries[car] < 100:
                car_batteries[car] += 1


# Atualiza o grafo e as capacidades das arestas
def update_graph():
    g = ChargingGraph()
    g.add_edge("T", "Charger1", 7.4)
    g.add_edge("T", "Charger2", 7.4)

    for car, battery in list(car_batteries.items())[:4]:
        capacity = get_capacity(battery)
        g.add_edge("Charger1", car, capacity)

    for car, battery in list(car_batteries.items())[4:]:
        capacity = get_capacity(battery)
        g.add_edge("Charger2", car, capacity)

    return g


def format_elapsed_time(elapsed_time):
    minutes, seconds = divmod(elapsed_time, 60)
    return f"Tempo decorrido: {minutes:02d}:{seconds:02d}"


app = dash.Dash(__name__)
start_time = time.time()

app.layout = html.Div(
    [
        dcc.Graph(id="live-graph"),
        dcc.Interval(
            id="graph-update",
            interval=10 * 1000,  # Atualiza a cada 10 segundos
            n_intervals=0,
        ),
        dcc.Interval(
            id="timer-update",
            interval=1 * 1000,  # Atualiza a cada 1 segundo
            n_intervals=0,
        ),
        html.Div(id="timer", style={"fontSize": 24, "marginTop": 20}),
    ]
)


@app.callback(Output("live-graph", "figure"), Input("graph-update", "n_intervals"))
def update_graph_live(n):
    g = update_graph()
    G_nx = g.create_networkx_graph()
    pos = hierarchy_pos(G_nx, "T")

    edge_trace = []
    edge_labels = {}
    for edge in G_nx.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=2, color="blue"),
            opacity=0.7,
        )
        edge_trace.append(trace)
        edge_labels[(edge[0], edge[1])] = f'{edge[2]["capacity"]} kW'

    node_trace = go.Scatter(
        x=[pos[node][0] for node in G_nx.nodes()],
        y=[pos[node][1] for node in G_nx.nodes()],
        mode="markers+text",
        text=[
            f"{node}: {car_batteries[node]}%" if node in car_batteries else node
            for node in G_nx.nodes()
        ],
        marker=dict(size=20, color="skyblue"),
        textposition="middle right",
    )

    fig = go.Figure(
        data=edge_trace + [node_trace],
        layout=go.Layout(
            title="Visualização do Grafo de Roteamento de Eletricidade com Níveis de Bateria",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            annotations=[
                dict(
                    x=(pos[edge[0]][0] + pos[edge[1]][0]) / 2,
                    y=(pos[edge[0]][1] + pos[edge[1]][1]) / 2,
                    text=edge_labels[(edge[0], edge[1])],
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="center",
                    ax=0,
                    ay=0,
                    bgcolor="white",
                    opacity=0.8,
                )
                for edge in G_nx.edges()
            ],
        ),
    )
    return fig


@app.callback(Output("timer", "children"), Input("timer-update", "n_intervals"))
def update_timer(n):
    elapsed_time = int(time.time() - start_time)
    return format_elapsed_time(elapsed_time)


# Layout hierárquico
def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    pos = {root: (xcenter, vert_loc)}
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph):
        raise TypeError("G must be a DiGraph")

    if not children:
        return pos

    dx = width / len(children)
    nextx = xcenter - width / 2 - dx / 2

    for child in children:
        nextx += dx
        pos.update(
            hierarchy_pos(
                G,
                child,
                width=dx,
                vert_gap=vert_gap,
                vert_loc=vert_loc - vert_gap,
                xcenter=nextx,
            )
        )
    return pos


if __name__ == "__main__":
    threading.Thread(target=charge_batteries, daemon=True).start()
    app.run_server(debug=True)
