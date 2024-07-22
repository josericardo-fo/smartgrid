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
        parent = defaultdict(lambda: None)
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
        return current_power + 63

    def calculate_worst_case_power_usage(self):
        worst_case_power = sum(
            7.0 if battery < 80 else 2.0
            for battery in car_batteries.values()
            if battery < 100
        )
        return worst_case_power + 63

    def adjust_charging_rates(self):
        total_power = self.get_current_power_usage()
        cars_slowed_down = set()

        if total_power > self.max_power:
            cars_charging_high = [
                (car, cap)
                for (charger, car), cap in self.capacity.items()
                if "Car" in car and cap == 7.0 and car_batteries[car] < 80
            ]
            cars_charging_high.sort(key=lambda x: car_batteries[x[0]], reverse=True)
            for car, cap in cars_charging_high:
                for charger in ["Charger1", "Charger2", "Charger3"]:
                    if (charger, car) in self.capacity:
                        self.capacity[(charger, car)] = 2.0
                        cars_slowed_down.add(car)
                total_power -= 5.0
                if total_power <= self.max_power:
                    break

        return cars_slowed_down

    def restore_charging_rates(self, cars_slowed_down):
        worst_case_power = self.calculate_worst_case_power_usage()

        if worst_case_power <= self.max_power:
            for (charger, car), capacity in list(self.capacity.items()):
                if "Car" in car and capacity == 2.0 and car_batteries[car] < 80:
                    self.capacity[(charger, car)] = 7.0
                    if car in cars_slowed_down:
                        cars_slowed_down.remove(car)


# Informações de bateria dos carros em porcentagem
car_batteries = {
    "Car1": 50,  # Carro 1 com 50% de bateria
    "Car2": 30,  # Carro 2 com 30% de bateria
    "Car3": 80,  # Carro 3 com 80% de bateria
    "Car4": 20,  # Carro 4 com 20% de bateria
    "Car5": 90,  # Carro 5 com 90% de bateria
    "Car6": 60,  # Carro 6 com 60% de bateria
    "Car7": 25,  # Carro 7 com 25% de bateria
    "Car8": 75,  # Carro 8 com 75% de bateria
    "Car9": 40,  # Carro 9 com 40% de bateria
}

# Variável para rastrear o tempo de carregamento
last_update_time = {car: time.time() for car in car_batteries}
slowed_down_cars = set()  # Carros que estão carregando mais lentamente


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
                increment = (
                    1
                    if current_time - last_update_time[car]
                    >= 2 * (2 if car in slowed_down_cars else 1)
                    else 0
                )
                car_batteries[car] = min(car_batteries[car] + increment, 100)
                last_update_time[car] = (
                    current_time if increment else last_update_time[car]
                )
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

    slowed_down_cars.update(g.adjust_charging_rates())

    for car in car_batteries:
        if car_batteries[car] >= 80:
            for charger in ["Charger1", "Charger2", "Charger3"]:
                if (charger, car) in g.capacity and g.capacity[(charger, car)] == 7.0:
                    g.capacity[(charger, car)] = 2.0
                    slowed_down_cars.add(car)

    g.restore_charging_rates(slowed_down_cars)
    return g, slowed_down_cars


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
    # Atualiza o gráfico de visualização do grafo
    g, cars_slowed_down = update_graph()
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
                    "orange"
                    if node in cars_slowed_down
                    else (
                        "green"
                        if node in car_batteries and car_batteries[node] == 100
                        else (
                            "yellow"
                            if node in car_batteries and car_batteries[node] >= 80
                            else "skyblue"
                        )
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
            ]
            + [
                dict(
                    x=1,
                    y=1,
                    xref="paper",
                    yref="paper",
                    text="<span style='color:green;'>⬤</span> Carro com 100% de bateria",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="left",
                ),
                dict(
                    x=1,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text="<span style='color:yellow;'>⬤</span> Carro com bateria >= 80%",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="left",
                ),
                dict(
                    x=1,
                    y=0.90,
                    xref="paper",
                    yref="paper",
                    text="<span style='color:orange;'>⬤</span> Carro que tiveram que carregar lentamente",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="left",
                ),
                dict(
                    x=1,
                    y=0.85,
                    xref="paper",
                    yref="paper",
                    text="<span style='color:skyblue;'>⬤</span> Carro com bateria < 80%",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="left",
                ),
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
