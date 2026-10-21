"""
subgraph.py

Subgraph isomorphism problem solver

TODO: What is the reference in the literature that has the algorithm implemented here?
Is it this?

Corneil, D.G. & Gotlieb, C.C. (1970). An efficient algorithm for graph isomorphism. *Journal of the Association for Computing Machinery*, 17, 51--64.
"""


def find_subgraph(G, H):
    """
    Finds an isomorphic mapping from graph H to a subgraph of graph G using
    backtracking with degree-based pruning and forward checking. This implementation is

    Parameters:
        G (list of list of int): The adjacency matrix of the target graph G.
            Should be an n x n matrix where G[i][j] == 1 if there is an edge from vertex i to j, else 0.
        H (list of list of int): The adjacency matrix of the pattern graph H.
            Should be an m x m matrix where H[i][j] == 1 if there is an edge from vertex i to j, else 0.

    Returns:
        dict or None: A dictionary mapping each vertex index of H (0 to m-1) to a vertex index of G (0 to n-1)
            such that the induced subgraph of G is isomorphic to H. Returns None if no such mapping exists.

    Notes:
        - The mapping is injective (no two H vertices map to the same G vertex).
        - Adjacency is preserved: for all i, j in H, H[i][j] == G[mapping[i]][mapping[j]].
        - Both G and H are expected to be undirected or directed graphs represented as adjacency matrices.
        - Time complexity: worst-case exponential in the number of vertices of H (O(n^m)),
          but with pruning strategies to improve practical performance.
    """
    n = len(G)
    m = len(H)

    if m > n:
        return None

    # Vertex properties for pruning
    G_degrees = [sum(G[i]) for i in range(n)]
    H_degrees = [sum(H[i]) for i in range(m)]

    # Create candidate sets: for each H vertex, which G vertices are possible
    candidates = []
    for h_v in range(m):
        h_deg = H_degrees[h_v]
        candidate_set = []
        for g_v in range(n):
            if G_degrees[g_v] >= h_deg:
                candidate_set.append(g_v)
        if not candidate_set:
            return None  # No valid candidates for this H vertex
        candidates.append(candidate_set)

    # Sort H vertices by number of candidates (ascending) for better pruning
    h_vertices = list(range(m))
    h_vertices.sort(key=lambda x: len(candidates[x]))

    mapping = {}
    used_in_G = [False] * n

    def is_valid_mapping(h_vertex, g_vertex):
        """
        Enhanced validity check with neighborhood comparison
        """
        for h_neighbor in range(m):
            if h_neighbor in mapping:
                g_neighbor = mapping[h_neighbor]
                if H[h_vertex][h_neighbor] != G[g_vertex][g_neighbor]:
                    return False
        return True

    def forward_check(h_vertex, g_vertex):
        """
        Check if mapping h_vertex to g_vertex would make future mappings impossible
        """
        # For each unmapped H vertex, check if it still has valid candidates
        for h_v in h_vertices:
            if h_v not in mapping and h_v != h_vertex:
                has_valid_candidate = False
                for g_v in candidates[h_v]:
                    if g_v != g_vertex and not used_in_G[g_v]:
                        # Check if this candidate is still valid given current mapping
                        valid = True
                        for h_mapped, g_mapped in mapping.items():
                            if H[h_v][h_mapped] == 1 and G[g_v][g_mapped] == 0:
                                valid = False
                                break
                        if valid and H[h_v][h_vertex] == 1 and G[g_v][g_vertex] == 0:
                            valid = False
                        if valid:
                            has_valid_candidate = True
                            break
                if not has_valid_candidate:
                    return False
        return True

    def backtrack(idx):
        """
        Backtracking search with pruning
        """
        if idx == m:
            return True

        h_vertex = h_vertices[idx]

        for g_vertex in candidates[h_vertex]:
            if not used_in_G[g_vertex] and is_valid_mapping(h_vertex, g_vertex):
                if forward_check(h_vertex, g_vertex):
                    mapping[h_vertex] = g_vertex
                    used_in_G[g_vertex] = True

                    if backtrack(idx + 1):
                        return True

                    del mapping[h_vertex]
                    used_in_G[g_vertex] = False

        return False

    if backtrack(0):
        return mapping
    else:
        return None


def verify_mapping(G, H, mapping):
    """
    Verifies whether the given mapping is a correct subgraph isomorphism from H to G.

    Args:
        G (list[list[int]]): The adjacency matrix of the target graph.
        H (list[list[int]]): The adjacency matrix of the pattern (subgraph).
        mapping (dict[int, int]): A mapping from vertices of H (keys) to vertices of G (values).

    Returns:
        bool: True if the mapping is a valid subgraph isomorphism (i.e., for all i, j in H, H[i][j] == G[mapping[i]][mapping[j]]), False otherwise.
    """
    m = len(H)
    for i in range(m):
        for j in range(m):
            if H[i][j] != G[mapping[i]][mapping[j]]:
                return False
    return True


def is_symmetric(matrix):
    """
    Checks if a matrix (represented as a list of lists) is symmetric.

    A matrix is symmetric if matrix[i][j] == matrix[j][i] for all valid i, j.

    Args:
        matrix: A list of lists representing a square matrix

    Returns:
        bool: True if the matrix is symmetric, False otherwise

    Raises:
        ValueError: If the matrix is not square or is empty
    """
    if not matrix:
        raise ValueError("Matrix is empty")

    n = len(matrix)

    # Check if matrix is square
    for i, row in enumerate(matrix):
        if len(row) != n:
            raise ValueError(f"Matrix is not square: row {i} has length {len(row)}, expected {n}")

    # Check symmetry
    for i in range(n):
        for j in range(i + 1, n):  # Only check upper triangle
            if matrix[i][j] != matrix[j][i]:
                return False

    return True


def main():
    """
    Demonstrates the use of the subgraph isomorphism solver.

    The example defines two undirected graphs using adjacency matrices:
    - G: a 5-node graph (adjacency matrix), representing the larger graph.
    - H: a 3-node graph (adjacency matrix), representing the smaller graph to search for as a subgraph.

    The function attempts to find a subgraph of G that is isomorphic to H.
    If a mapping is found, it prints the mapping from H's nodes to G's nodes and verifies the mapping.
    Otherwise, it reports that no subgraph was found.
    """
    G = [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0],
    ]

    H = [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]

    assert is_symmetric(G), "Graph G should be undirected (symmetric adjacency matrix)"
    assert is_symmetric(H), "Graph H should be undirected (symmetric adjacency matrix)"

    mapping = find_subgraph(G, H)
    if mapping:
        print(f"Subgraph found! Mapping: {mapping}")
        print(f"Verification: {verify_mapping(G, H, mapping)}")
    else:
        print("Subgraph not found")


if __name__ == "__main__":
    main()
