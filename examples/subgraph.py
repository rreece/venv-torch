"""
subgraph.py

Subgraph isomorphism by backtracking search with forward checking,
following Haralick & Elliott (1980).

Two versions of the problem are supported, selected by the induced flag. Both
look for an injective mapping of the vertices of H into G:

- induced=True (default): induced subgraph isomorphism. Edges of H must map
  to edges of G, and non-edges of H must map to non-edges of G. That is, H is
  isomorphic to the subgraph of G induced by the image vertices: those
  vertices together with all of G's edges between them. Other vertices of G,
  and edges to them, are ignored.
- induced=False: non-induced subgraph isomorphism, also called subgraph
  monomorphism. Edges of H must map to edges of G, but G may have extra edges
  between the image vertices. This is the problem Ullmann (1976) solves, and
  the right one when every connection H needs must exist in G but unused
  connections in G are harmless, as when placing a netlist or a circuit of
  qubits onto hardware with fixed links.

An induced match is also a non-induced match, but not the other way round.
For example, a path a-b-c is a non-induced subgraph of a triangle, but not an
induced one, because the triangle has the extra edge a-c. When H is complete,
it has no non-edges, and the two versions coincide.

Each unmapped H vertex keeps a domain of Each unmapped H vertex keeps a domain of
candidate G vertices, stored as a bitset. Initial domains use the degree filter
of Ullmann (1976). After each assignment, forward checking removes the
candidates that are inconsistent with it, and backtracks if any domain becomes
empty, so every remaining candidate is consistent with the whole partial
mapping. The next H vertex is chosen dynamically as the one with the fewest
candidates left (fail first), with ties going to the vertex of highest degree,
as in Ullmann's ordering.

Optionally, after each forward check, Ullmann's refinement procedure is run to
a fixpoint: a candidate x for an unmapped vertex u is removed unless every
other unmapped vertex w still has a candidate compatible with u -> x. With
refine="neighbors", w ranges over the H-neighbours of u, as in Ullmann's
condition (2). With refine="full", w ranges over all unmapped vertices; this
is the full looking ahead of Haralick & Elliott. For induced=False the two are
equivalent, apart from injectivity, since only edges of H constrain the
mapping. For induced=True, "full" also uses non-edges of H as constraints.

References:

- Corneil, D.G. & Gotlieb, C.C. (1970). An efficient algorithm for graph isomorphism. *Journal of the Association for Computing Machinery*, 17, 51--64.
- Ullmann, J.R. (1976). An algorithm for subgraph isomorphism. *Journal of the Association for Computing Machinery*, 23, 31--42.
- Haralick, R.M., & Elliott, G.L. (1980). Increasing tree search efficiency for constraint satisfaction problems. *Artificial Intelligence*, 14, 263--313.
- Cordella, L.P., Foggia, P., Sansone, C., & Vento, M. (2004). A (sub) graph isomorphism algorithm for matching large graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 26, 1367--1372.
"""


def find_subgraph(G, H, induced=True, refine=None, stats=None):
    """
    Finds an isomorphic mapping from graph H to a subgraph of graph G using
    backtracking with degree-based pruning and forward checking. By default the
    subgraph must be induced; see the module docstring for the difference, and
    for how this relates to the references.

    Parameters:
        G (list of list of int): The adjacency matrix of the target graph G.
            Should be an n x n matrix where G[i][j] == 1 if there is an edge from vertex i to j, else 0.
        H (list of list of int): The adjacency matrix of the pattern graph H.
            Should be an m x m matrix where H[i][j] == 1 if there is an edge from vertex i to j, else 0.
        induced (bool, optional): True (default) for induced subgraph isomorphism, where edges and
            non-edges of H must both be preserved. False for non-induced subgraph isomorphism
            (monomorphism), where only edges of H must be preserved and G may have extra edges.
        refine (str, optional): None (default) for forward checking only, "neighbors" to also run
            Ullmann's refinement over H-neighbours, or "full" to run it over all unmapped vertices.
        stats (dict, optional): If given, search counters are accumulated into it:
            "nodes" (search tree nodes visited) and "checks" (forward checks of candidate assignments).

    Returns:
        dict or None: A dictionary mapping each vertex index of H (0 to m-1) to a vertex index of G (0 to n-1)
            such that H is isomorphic to a subgraph of G on the image vertices (the induced subgraph,
            if induced is True). Returns None if no such mapping exists.

    Notes:
        - The mapping is injective (no two H vertices map to the same G vertex).
        - Adjacency is preserved: for all i, j in H, H[i][j] == G[mapping[i]][mapping[j]] if induced,
          or H[i][j] == 1 implies G[mapping[i]][mapping[j]] == 1 if not induced.
        - Both G and H are expected to be undirected or directed graphs represented as adjacency matrices.
        - Time complexity: worst-case exponential in the number of vertices of H (O(n^m)),
          but with pruning strategies to improve practical performance.
    """
    n = len(G)
    m = len(H)

    if refine not in (None, "neighbors", "full"):
        raise ValueError(f"refine must be None, 'neighbors' or 'full', not {refine!r}")

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

    # compat[x][r] is the bitset of G vertices y that can be assigned to an H
    # vertex w, given that an H vertex u is assigned x, where r = rel[u][w]
    # encodes the edges between u and w. An edge of H must map to an edge of
    # G. A non-edge of H must map to a non-edge of G if induced, and is
    # unconstrained otherwise. Always y != x, so the mapping is injective.
    all_G = (1 << n) - 1
    compat = []
    for x in range(n):
        not_in = ~G_in[x] if induced else all_G
        not_out = ~G_out[x] if induced else all_G
        masks = []
        for w_to_u in (0, 1):
            for u_to_w in (0, 1):
                in_part = G_in[x] if w_to_u else not_in
                out_part = G_out[x] if u_to_w else not_out
                masks.append(in_part & out_part & ~(1 << x) & all_G)
        compat.append(masks)
    rel = [[2 * H[w][u] + H[u][w] for w in range(m)] for u in range(m)]

    # Initial domains: Ullmann's degree filter, plus self-loops, which must
    # match if induced, and must exist in G where H has one otherwise
    domains = {}
    for h in range(m):
        h_out = sum(H[h])
        h_in = sum(H[i][h] for i in range(m))
        domain = 0
        for g in range(n):
            loop_ok = G[g][g] == H[h][h] if induced else G[g][g] >= H[h][h]
            if G_out[g].bit_count() >= h_out and G_in[g].bit_count() >= h_in and loop_ok:
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
        masks = compat[g]
        filtered = {}
        for u, domain in future.items():
            domain &= masks[rel[h][u]]
            if not domain:
                return None
            filtered[u] = domain
        return filtered

    # Vertices whose domains support each other in refinement
    if refine == "neighbors":
        partners = [[w for w in range(m) if w != u and (H[u][w] or H[w][u])] for u in range(m)]
    else:
        partners = [[w for w in range(m) if w != u] for u in range(m)]

    def refine_domains(domains):
        """
        Ullmann's refinement procedure: remove each candidate x for u unless
        every partner w of u has a candidate compatible with u -> x, and repeat
        until nothing changes. Modifies domains in place and returns it, or
        returns None if any domain becomes empty.
        """
        changed = True
        while changed:
            changed = False
            for u, domain in domains.items():
                kept = domain
                rest = domain
                while rest:
                    low = rest & -rest
                    rest ^= low
                    masks = compat[low.bit_length() - 1]
                    for w in partners[u]:
                        if w in domains and not domains[w] & masks[rel[u][w]]:
                            kept ^= low
                            break
                if kept != domain:
                    if not kept:
                        return None
                    domains[u] = kept
                    changed = True
        return domains

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
            if filtered is not None and refine:
                filtered = refine_domains(filtered)
            if filtered is not None:
                mapping[h] = g
                if backtrack(filtered):
                    return True
                del mapping[h]

        return False

    # Ullmann also refines the initial domains before the search
    if refine and refine_domains(domains) is None:
        return None

    if backtrack(domains):
        return mapping
    else:
        return None


def verify_mapping(G, H, mapping, induced=True):
    """
    Verifies whether the given mapping is a correct subgraph isomorphism from H to G.

    Args:
        G (list[list[int]]): The adjacency matrix of the target graph.
        H (list[list[int]]): The adjacency matrix of the pattern (subgraph).
        mapping (dict[int, int]): A mapping from vertices of H (keys) to vertices of G (values).
        induced (bool): True (default) to check induced subgraph isomorphism, False to check
            non-induced subgraph isomorphism (monomorphism). See find_subgraph.

    Returns:
        bool: True if the mapping is injective and, for all i, j in H, H[i][j] == G[mapping[i]][mapping[j]]
            (if induced) or H[i][j] == 1 implies G[mapping[i]][mapping[j]] == 1 (if not induced).
            False otherwise.
    """
    m = len(H)
    if len(set(mapping[i] for i in range(m))) != m:
        return False
    for i in range(m):
        for j in range(m):
            g_edge = G[mapping[i]][mapping[j]]
            if (H[i][j] != g_edge) if induced else (H[i][j] > g_edge):
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


def from_edges(n, edges):
    """
    Returns the adjacency matrix of an undirected graph with n vertices and the given edges.

    Args:
        n (int): The number of vertices.
        edges (list[tuple[int, int]]): The edges, as pairs of vertex indices.

    Returns:
        list[list[int]]: An n x n symmetric adjacency matrix.
    """
    A = [[0] * n for _ in range(n)]
    for a, b in edges:
        A[a][b] = A[b][a] = 1
    return A


def main():
    """
    Demonstrates the difference between induced and non-induced subgraph isomorphism,
    as a toy placement problem.

    G is a 2 x 3 mesh of hardware cells, with one extra diagonal link 0-4 in the left square:

        0 - 1 - 2
        | \\ |   |
        3 - 4 - 5

    The patterns to place are:
    - ring: a ring of four operations, a - b - c - d - a.
    - ring+diagonal: the same ring with an extra link a - c.

    For the ring, both squares of the mesh contain the four links it needs, so the
    non-induced search may use either square; it does not matter that the left square
    has a diagonal link the ring does not use. The induced search must use the right
    square {1, 2, 4, 5}, because in the left square the diagonal is an extra edge between
    the chosen cells. For placement, where unused links are harmless, the non-induced
    version is the right question.

    For ring+diagonal, only the left square has the extra link, so both versions must
    use the left square {0, 1, 3, 4}.

    For each pattern and version, the function prints the mapping from pattern vertices
    to cells, the set of cells used, and whether verify_mapping accepts the mapping.
    """
    G = from_edges(6, [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5), (0, 4)])
    patterns = {
        "ring": from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        "ring+diagonal": from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),
    }

    assert is_symmetric(G), "Graph G should be undirected (symmetric adjacency matrix)"

    for name, H in patterns.items():
        assert is_symmetric(H), f"Pattern {name} should be undirected (symmetric adjacency matrix)"
        for induced in (False, True):
            label = f"{name}, {'induced' if induced else 'non-induced'}"
            mapping = find_subgraph(G, H, induced=induced)
            if mapping is None:
                print(f"{label}: not found")
            else:
                cells = sorted(mapping.values())
                ok = verify_mapping(G, H, mapping, induced=induced)
                print(f"{label}: mapping {dict(sorted(mapping.items()))}, cells {cells}, verified {ok}")


if __name__ == "__main__":
    main()
