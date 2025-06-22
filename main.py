from graph import Graph

def main():
    grafo = Graph(use_matrix=True)
    grafo.load_from_file("entrada.txt")

    grafo.export_representation("grafo.txt")
    grafo.export_graph_info("saida_info.txt")
    grafo.export_dfs_order("saida_dfs.txt")

    dfs_result = grafo.dfs_with_levels(start=1)
    grafo.export_search_tree(dfs_result, "dfs_tree.txt", method="DFS")

    bfs_result = grafo.bfs_with_levels(start=1)
    grafo.export_search_tree(bfs_result, "bfs_tree.txt", method="BFS")

    grafo.export_connected_components("componentes.txt")
    grafo.export_shortest_paths(start=1, filepath="caminhos_minimos.txt")

if __name__ == "__main__":
    main()
