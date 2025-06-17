from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque

class Graph:
    def __init__(self, use_matrix: bool = False):
        self.use_matrix = use_matrix
        self.num_vertices = 0
        self.edge_count = 0
        self.adj_list: Dict[int, List[int]] = defaultdict(list)
        self.adj_matrix: Optional[List[List[int]]] = None
        self.degrees: Dict[int, int] = defaultdict(int)

    def load_from_file(self, filepath: str):
        with open(filepath, 'r') as file:
            lines = file.readlines()
            self.num_vertices = int(lines[0].strip())
            if self.use_matrix:
                self.adj_matrix = [[0] * self.num_vertices for _ in range(self.num_vertices)]
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                u, v = int(parts[0]), int(parts[1])
                self.add_edge(u, v)

    def add_edge(self, u: int, v: int):
        if self.use_matrix:
            self.adj_matrix[u-1][v-1] = 1
            self.adj_matrix[v-1][u-1] = 1
        else:
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)
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
                file.write("# Matriz de Adjacência\n")
                for row in self.adj_matrix:
                    line = ' '.join(map(str, row))
                    file.write(f"{line}\n")
            else:
                file.write("# Lista de Adjacência\n")
                for vertex in sorted(self.adj_list.keys()):
                    neighbors = ' '.join(map(str, sorted(self.adj_list[vertex])))
                    file.write(f"{vertex}: {neighbors}\n")


    def get_neighbors(self, v: int):
        if self.use_matrix:
            return [i + 1 for i, has_edge in enumerate(self.adj_matrix[v - 1]) if has_edge]
        else:
            return self.adj_list[v]
        
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
