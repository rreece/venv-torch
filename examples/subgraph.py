"""
subgraph.py

Induced subgraph isomorphism by backtracking search with forward checking,
following Haralick & Elliott (1980). Each unmapped H vertex keeps a domain of
candidate G vertices, stored as a bitset. Initial domains use the degree filter
of Ullmann (1976). After each assignment, forward checking removes the
candidates that are inconsistent with it, and backtracks if any domain becomes
empty, so every remaining candidate is consistent with the whole partial
mapping. The next H vertex is chosen dynamically as the one with the fewest
candidates left (fail first), with ties going to the vertex of highest degree,
as in Ullmann's ordering. Ullmann's refinement procedure, which is a full
look-ahead, is not implemented.

References:

- Corneil, D.G. & Gotlieb, C.C. (1970). An efficient algorithm for graph isomorphism. *Journal of the Association for Computing Machinery*, 17, 51--64.
- Ullmann, J.R. (1976). An algorithm for subgraph isomorphism. *Journal of the Association for Computing Machinery*, 23, 31--42.
- Haralick, R.M., & Elliott, G.L. (1980). Increasing tree search efficiency for constraint satisfaction problems. *Artificial Intelligence*, 14, 263--313.
- Cordella, L.P., Foggia, P., Sansone, C., & Vento, M. (2004). A (sub) graph isomorphism algorithm for matching large graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 26, 1367--1372.
"""


def find_subgraph(G, H, stats=None):
    """
    Finds an isomorphic mapping from graph H to an induced subgraph of graph G using
    backtracking with degree-based pruning and forward checking. See the module
    docstring for how this relates to the references.

    Parameters:
        G (list of list of int): The adjacency matrix of the target graph G.
            Should be an n x n matrix where G[i][j] == 1 if there is an edge from vertex i to j, else 0.
        H (list of list of int): The adjacency matrix of the pattern graph H.
            Should be an m x m matrix where H[i][j] == 1 if there is an edge from vertex i to j, else 0.
        stats (dict, optional): If given, search counters are accumulated into it:
            "nodes" (search tree nodes visited) and "checks" (forward checks of candidate assignments).

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

    if stats is None:
        stats = {}
    stats.setdefault("nodes", 0)
    stats.setdefault("checks", 0)

    if m > n:
        return None

    # Bitsets (Python ints) of out- and in-neighbours of each G vertex, so that
    # domains can be filtered bit-parallel as in Haralick & Elliott (1980)
    G_out = [sum(1 << j for j in range(n) if G[i][j]) for i in range(n)]
    G_in = [sum(1 << i for i in range(n) if G[i][j]) for j in range(n)]

    # Initial domains: Ullmann's degree filter, plus matching self-loops
    domains = {}
    for h in range(m):
        h_out = sum(H[h])
        h_in = sum(H[i][h] for i in range(m))
        domain = 0
        for g in range(n):
            if (G_out[g].bit_count() >= h_out and G_in[g].bit_count() >= h_in
                    and G[g][g] == H[h][h]):
                domain |= 1 << g
        if not domain:
            return None  # No valid candidates for this H vertex
        domains[h] = domain

    # Total (out + in) degree of each H vertex, for breaking ties in fail first
    H_deg = [sum(H[u]) + sum(row[u] for row in H) for u in range(m)]

    mapping = {}

    def forward_check(h, g, future):
        """
        Filter the domains of the future H vertices given the assignment h -> g.
        Returns the filtered domains, or None if any domain becomes empty.
        """
        filtered = {}
        for u, domain in future.items():
            domain &= ~(1 << g)  # injective
            # Induced: edges and non-edges between u and h must both match
            domain &= G_in[g] if H[u][h] else ~G_in[g]
            domain &= G_out[g] if H[h][u] else ~G_out[g]
            if not domain:
                return None
            filtered[u] = domain
        return filtered

    def backtrack(domains):
        """
        Backtracking search. Every candidate left in domains is consistent
        with the current partial mapping.
        """
        stats["nodes"] += 1
        if not domains:
            return True

        # Fail first: assign the H vertex with the fewest candidates left,
        # breaking ties by highest degree as in Ullmann's ordering
        h = min(domains, key=lambda u: (domains[u].bit_count(), -H_deg[u]))
        future = {u: d for u, d in domains.items() if u != h}

        domain = domains[h]
        while domain:
            low = domain & -domain
            domain ^= low
            g = low.bit_length() - 1

            stats["checks"] += 1
            filtered = forward_check(h, g, future)
            if filtered is not None:
                mapping[h] = g
                if backtrack(filtered):
                    return True
                del mapping[h]

        return False

    if backtrack(domains):
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
