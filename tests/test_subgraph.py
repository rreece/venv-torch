"""
Test examples/subgraph.py
"""

import itertools
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from subgraph import find_subgraph, verify_mapping  # noqa: E402


def brute_force_exists(G, H):
    n, m = len(G), len(H)
    for perm in itertools.permutations(range(n), m):
        if verify_mapping(G, H, dict(enumerate(perm))):
            return True
    return False


def random_graph(n, p, directed=False, loops=False, rng=random):
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j and not loops:
                continue
            if not directed and j < i:
                continue
            if rng.random() < p:
                A[i][j] = 1
                if not directed:
                    A[j][i] = 1
    return A


def test_triangle_in_undirected_graph():
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
    mapping = find_subgraph(G, H)
    assert mapping is not None
    assert verify_mapping(G, H, mapping)
    assert set(mapping.values()) == {2, 3, 4}


def test_no_triangle_in_path():
    G = [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ]
    H = [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]
    assert find_subgraph(G, H) is None


def test_directed_edge_direction():
    # H: 0 -> 1; G: 0 -> 2 only. Must map onto the 0 -> 2 edge.
    H = [[0, 1], [0, 0]]
    G = [[0, 0, 1], [0, 0, 0], [0, 0, 0]]
    mapping = find_subgraph(G, H)
    assert mapping == {0: 0, 1: 2}
    assert verify_mapping(G, H, mapping)


def test_directed_no_match():
    # H: 2-cycle 0 <-> 1; G: a directed path, no 2-cycle
    H = [[0, 1], [1, 0]]
    G = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    assert find_subgraph(G, H) is None


def test_self_loop_must_match():
    H = [[1]]
    G = [[0, 1], [0, 0]]
    assert find_subgraph(G, H) is None

    G = [[0, 1], [0, 1]]
    mapping = find_subgraph(G, H)
    assert mapping == {0: 1}
    assert verify_mapping(G, H, mapping)


def test_no_self_loop_in_H():
    # Induced: H vertex without a loop can't map to a G vertex with one
    H = [[0]]
    G = [[1]]
    assert find_subgraph(G, H) is None


def test_pattern_larger_than_target():
    assert find_subgraph([[0]], [[0, 1], [1, 0]]) is None


def test_random_against_brute_force():
    rng = random.Random(0)
    for _ in range(300):
        directed = rng.random() < 0.5
        loops = rng.random() < 0.3
        n = rng.randint(1, 6)
        m = rng.randint(1, min(n, 4))
        G = random_graph(n, rng.random(), directed, loops, rng)
        H = random_graph(m, rng.random(), directed, loops, rng)
        mapping = find_subgraph(G, H)
        if mapping is None:
            assert not brute_force_exists(G, H), (G, H)
        else:
            assert verify_mapping(G, H, mapping), (G, H, mapping)
