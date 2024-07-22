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
        self.graph = defaultdict(
            list
        )  # Grafo representado como uma lista de adjacências
        self.capacity = {}  # Dicionário para armazenar as capacidades das arestas
        self.max_power = 102.5  # Capacidade máxima de carregamento (em kW)

    def add_edge(self, u, v, cap):
        # Adiciona uma aresta entre os nós u e v com capacidade cap
        self.graph[u].append(v)
        self.graph[v].append(u)
        self.capacity[(u, v)] = cap
        self.capacity[(v, u)] = 0

    def bfs(self, source, sink, parent):
        # Busca em largura para encontrar um caminho aumentante no grafo residual
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
        # Algoritmo de fluxo máximo de Edmonds-Karp
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
        # Cria um grafo NetworkX para visualização
        G = nx.DiGraph()
        for u in self.graph:
            for v in self.graph[u]:
                if (u, v) in self.capacity and self.capacity[(u, v)] > 0:
                    G.add_edge(u, v, capacity=self.capacity[(u, v)])
        return G

    def get_current_power_usage(self):
        # Calcula a potência atual utilizada no carregamento dos carros
        current_power = 0
        for (u, v), capacity in self.capacity.items():
            if "Charger" in u and "Car" in v and car_batteries[v] < 100:
                current_power += capacity
        return (
            current_power + 63
        )  # Potência base de 63 kW para os carregadores de 21 kW cada


# Informações de bateria dos carros em porcentagem
car_batteries = {
    "Car1": 50,  # Carro 1 com 50% de bateria
    "Car2": 30,  # Carro 2 com 30% de bateria
    "Car3": 80,  # Carro 3 com 85% de bateria
    "Car4": 20,  # Carro 4 com 20% de bateria
    "Car5": 90,  # Carro 5 com 95% de bateria
    "Car6": 60,  # Carro 6 com 60% de bateria
    "Car7": 25,  # Carro 7 com 25% de bateria
    "Car8": 75,  # Carro 8 com 75% de bateria
    "Car9": 40,  # Carro 9 com 40% de bateria
}

# Variável para rastrear o tempo de carregamento
last_update_time = {car: time.time() for car in car_batteries}


def get_capacity(battery_percentage):
    # Define a capacidade de carregamento baseada na porcentagem da bateria
    if battery_percentage < 80:
        return 7.0  # Capacidade alta para baterias < 80% (em kW)
    elif battery_percentage < 100:
        return 2.0  # Capacidade baixa para baterias >= 80% (em kW)
    else:
        return 0.0  # Capacidade nula para baterias cheias


def charge_batteries():
    # Simula o processo de carregamento das baterias dos carros
    while True:
        current_time = time.time()
        for car in car_batteries:
            if car_batteries[car] < 100:
                if car_batteries[car] < 80:
                    if (
                        current_time - last_update_time[car] >= 2
                    ):  # Carrega 1% a cada 2 segundos
                        car_batteries[car] += 1
                        last_update_time[car] = current_time
                else:
                    if (
                        current_time - last_update_time[car] >= 4
                    ):  # Carrega 1% a cada 4 segundos
                        car_batteries[car] += 1
                        last_update_time[car] = current_time
                car_batteries[car] = min(
                    car_batteries[car], 100
                )  # Garantir que não passe de 100%
        time.sleep(1)


def update_graph():
    # Atualiza o grafo e as capacidades das arestas
    g = ChargingGraph()
    g.add_edge("T", "Charger1", 21)
    g.add_edge("T", "Charger2", 21)
    g.add_edge("T", "Charger3", 21)

    # Distribuir os carros entre os três carregadores
    chargers = ["Charger1", "Charger2", "Charger3"]
    for i, (car, battery) in enumerate(car_batteries.items()):
        capacity = get_capacity(battery)  # Define a capacidade de carregamento
        g.add_edge(chargers[i % 3], car, capacity)

    return g


def format_elapsed_time(elapsed_time):
    # Formata o tempo decorrido em minutos e segundos
    minutes, seconds = divmod(elapsed_time, 60)
    return f"Tempo decorrido: {minutes:02d}:{seconds:02d}"


app = dash.Dash(__name__)
start_time = time.time()

app.layout = html.Div(
    [
        dcc.Graph(id="live-graph"),
        dcc.Interval(
            id="graph-update",
            interval=1000,  # Atualiza a cada 1 segundo
            n_intervals=0,
        ),
        dcc.Interval(
            id="timer-update",
            interval=1000,  # Atualiza a cada 1 segundo
            n_intervals=0,
        ),
        html.Div(id="timer", style={"fontSize": 24, "marginTop": 20}),
        html.Div(id="power-usage", style={"fontSize": 24, "marginTop": 20}),
    ]
)


@app.callback(
    [Output("live-graph", "figure"), Output("power-usage", "children")],
    [Input("graph-update", "n_intervals")],
)
def update_graph_live(n):
    # Atualiza o grafo e a visualização do uso de energia em tempo real
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
        marker=dict(
            size=20,
            color=[
                (
                    "green"
                    if node in car_batteries and car_batteries[node] == 100
                    else (
                        "orange"
                        if node in car_batteries and car_batteries[node] > 80
                        else "skyblue"
                    )
                )
                for node in G_nx.nodes()
            ],
        ),
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

    current_power_usage = g.get_current_power_usage()
    power_usage_text = f"Uso de energia atual: {current_power_usage}/{g.max_power} kW"

    return fig, power_usage_text


@app.callback(Output("timer", "children"), Input("timer-update", "n_intervals"))
def update_timer(n):
    # Atualiza o tempo decorrido na interface
    elapsed_time = int(time.time() - start_time)
    return format_elapsed_time(elapsed_time)


def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    # Calcula a posição hierárquica dos nós para visualização
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
    # Inicia a simulação de carregamento das baterias em uma thread separada
    threading.Thread(target=charge_batteries, daemon=True).start()
    # Inicia o servidor Dash para a visualização em tempo real
    app.run_server(debug=True)
