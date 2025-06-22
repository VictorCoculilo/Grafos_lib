from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
import heapq

class Graph:
    def __init__(self, use_matrix: bool = False):
        self.use_matrix = use_matrix
        self.num_vertices = 0
        self.edge_count = 0
        self.adj_list: Dict[int, List[Tuple[int, float]]] = defaultdict(list)  # (vizinho, peso)
        self.adj_matrix: Optional[List[List[Optional[float]]]] = None
        self.degrees: Dict[int, int] = defaultdict(int)

    def load_from_file(self, filepath: str):
        with open(filepath, 'r') as file:
            lines = file.readlines()
            self.num_vertices = int(lines[0].strip())
            if self.use_matrix:
                self.adj_matrix = [[None] * self.num_vertices for _ in range(self.num_vertices)]
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                u, v = int(parts[0]), int(parts[1])
                weight = float(parts[2]) if len(parts) > 2 else 1.0
                self.add_edge(u, v, weight)

    def add_edge(self, u: int, v: int, weight: float = 1.0):
        if self.use_matrix:
            self.adj_matrix[u-1][v-1] = weight
            self.adj_matrix[v-1][u-1] = weight
        else:
            self.adj_list[u].append((v, weight))
            self.adj_list[v].append((u, weight))
        self.edge_count += 1
        self.degrees[u] += 1
        self.degrees[v] += 1

    def export_graph_info(self, filepath: str):
        with open(filepath, 'w') as file:
            file.write(f"# n = {self.num_vertices}\n")
            file.write(f"# m = {self.edge_count}\n")
            d_medio = sum(self.degrees.values()) / self.num_vertices
            file.write(f"# d_medio = {round(d_medio, 2)}\n")
            for v in sorted(self.degrees.keys()):
                p = round(self.degrees[v] / (2 * self.edge_count), 2)
                file.write(f"{v} {p:.2f}\n")

    def export_representation(self, filepath: str):
        with open(filepath, 'w') as file:
            if self.use_matrix:
                file.write("# Matriz de Adjacência com Pesos\n")
                for row in self.adj_matrix:
                    line = ' '.join(str(val) if val is not None else '0.0' for val in row)
                    file.write(f"{line}\n")
            else:
                file.write("# Lista de Adjacência com Pesos\n")
                for vertex in sorted(self.adj_list.keys()):
                    neighbors = ' '.join(f"{v}({w})" for v, w in sorted(self.adj_list[vertex]))
                    file.write(f"{vertex}: {neighbors}\n")

    def get_neighbors(self, v: int) -> List[int]:
        if self.use_matrix:
            return [i + 1 for i, weight in enumerate(self.adj_matrix[v - 1]) if weight is not None]
        else:
            return [neighbor for neighbor, _ in self.adj_list[v]]

    def get_edge_weight(self, u: int, v: int) -> float:
        if self.use_matrix:
            weight = self.adj_matrix[u - 1][v - 1]
            return weight if weight is not None else float('inf')
        else:
            for neighbor, weight in self.adj_list[u]:
                if neighbor == v:
                    return weight
            return float('inf')

    def dfs_descending_order(self) -> List[int]:
        visited = set()
        finish_order = []

        def dfs(v: int):
            visited.add(v)
            for neighbor in self.get_neighbors(v):
                if neighbor not in visited:
                    dfs(neighbor)
            finish_order.append(v)

        for vertex in range(1, self.num_vertices + 1):
            if vertex not in visited:
                dfs(vertex)

        return finish_order[::-1]

    def export_dfs_order(self, filepath: str):
        order = self.dfs_descending_order()
        with open(filepath, 'w') as file:
            file.write("# Ordem decrescente por término (DFS)\n")
            for vertex in order:
                file.write(f"{vertex}\n")

    def bfs_with_levels(self, start: int) -> List[Tuple[int, Optional[int], int]]:
        visited = set()
        level = {start: 0}
        parent = {start: None}
        queue = deque([start])
        result = []

        while queue:
            v = queue.popleft()
            visited.add(v)
            result.append((v, parent[v], level[v]))
            for neighbor in self.get_neighbors(v):
                if neighbor not in visited and neighbor not in queue:
                    parent[neighbor] = v
                    level[neighbor] = level[v] + 1
                    queue.append(neighbor)
        return result

    def dfs_with_levels(self, start: int) -> List[Tuple[int, Optional[int], int]]:
        visited = set()
        result = []

        def dfs(v: int, parent_v: Optional[int], lvl: int):
            visited.add(v)
            result.append((v, parent_v, lvl))
            for neighbor in self.get_neighbors(v):
                if neighbor not in visited:
                    dfs(neighbor, v, lvl + 1)

        dfs(start, None, 0)
        return result

    def export_search_tree(self, result: List[Tuple[int, Optional[int], int]], filepath: str, method: str):
        with open(filepath, 'w') as file:
            file.write(f"# Árvore de busca gerada por {method}\n")
            for vertex, parent, lvl in result:
                file.write(f"Vértice: {vertex} | Pai: {parent if parent else '-'} | Nível: {lvl}\n")

    def connected_components(self) -> List[List[int]]:
        visited = set()
        components = []

        def dfs(v: int, current_component: List[int]):
            visited.add(v)
            current_component.append(v)
            for neighbor in self.get_neighbors(v):
                if neighbor not in visited:
                    dfs(neighbor, current_component)

        for vertex in range(1, self.num_vertices + 1):
            if vertex not in visited:
                component = []
                dfs(vertex, component)
                components.append(sorted(component))

        components.sort(key=len, reverse=True)
        return components

    def export_connected_components(self, filepath: str):
        components = self.connected_components()
        with open(filepath, 'w') as file:
            file.write(f"# Número de componentes conexos: {len(components)}\n")
            for i, comp in enumerate(components, 1):
                file.write(f"Componente {i} (tamanho {len(comp)}): {' '.join(map(str, comp))}\n")

    def has_negative_weights(self) -> bool:
        if self.use_matrix:
            return any(weight is not None and weight < 0
                       for row in self.adj_matrix for weight in row)
        else:
            return any(weight < 0 for neighbors in self.adj_list.values() for _, weight in neighbors)

    def has_weights(self) -> bool:
        if self.use_matrix:
            return any(weight is not None and weight != 1
                       for row in self.adj_matrix for weight in row)
        else:
            return any(weight != 1 for neighbors in self.adj_list.values() for _, weight in neighbors)

    def _dijkstra(self, start: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        dist = {v: float('inf') for v in range(1, self.num_vertices + 1)}
        prev = {v: None for v in range(1, self.num_vertices + 1)}
        dist[start] = 0

        heap = [(0, start)]
        while heap:
            current_dist, u = heapq.heappop(heap)
            if current_dist > dist[u]:
                continue
            for v in self.get_neighbors(u):
                weight = self.get_edge_weight(u, v)
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    prev[v] = u
                    heapq.heappush(heap, (dist[v], v))
        return dist, prev

    def _bfs_shortest_path(self, start: int) -> Tuple[Dict[int, int], Dict[int, Optional[int]]]:
        dist = {start: 0}
        prev = {start: None}
        queue = deque([start])

        while queue:
            u = queue.popleft()
            for v in self.get_neighbors(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    queue.append(v)
        return dist, prev

    def shortest_path(self, start: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        if self.use_matrix or self.has_weights():
            if self.has_negative_weights():
                raise ValueError("O grafo contém pesos negativos. Dijkstra não é aplicável.")
            return self._dijkstra(start)
        else:
            return self._bfs_shortest_path(start)

    def reconstruct_path(self, prev: Dict[int, Optional[int]], end: int) -> List[int]:
        path = []
        while end is not None:
            path.append(end)
            end = prev[end]
        return path[::-1]

    def export_shortest_paths(self, start: int, filepath: str):
        try:
            dist, prev = self.shortest_path(start)
            with open(filepath, 'w') as file:
                file.write(f"Caminhos mínimos a partir do vértice {start}\n")
                for v in range(1, self.num_vertices + 1):
                    if v == start:
                        continue
                    path = self.reconstruct_path(prev, v)
                    if dist[v] == float('inf'):
                        file.write(f"{start} -> {v}: sem caminho\n")
                    else:
                        caminho_str = " -> ".join(map(str, path))
                        file.write(f"{start} -> {v}: {caminho_str} (distância = {dist[v]})\n")
        except ValueError as e:
            with open(filepath, 'w') as file:
                file.write(f"Erro: {e}\n")
