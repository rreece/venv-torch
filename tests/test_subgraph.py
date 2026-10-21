"""
Test examples/subgraph.py
"""

import itertools
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from subgraph import find_subgraph, verify_mapping  # noqa: E402

REFINE = [None, "neighbors", "full"]


def brute_force_exists(G, H, induced=True):
    n, m = len(G), len(H)
    for perm in itertools.permutations(range(n), m):
        if verify_mapping(G, H, dict(enumerate(perm)), induced=induced):
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


@pytest.mark.parametrize("refine", REFINE)
def test_triangle_in_undirected_graph(refine):
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
    mapping = find_subgraph(G, H, refine=refine)
    assert mapping is not None
    assert verify_mapping(G, H, mapping)
    assert set(mapping.values()) == {2, 3, 4}


@pytest.mark.parametrize("refine", REFINE)
def test_no_triangle_in_path(refine):
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
    assert find_subgraph(G, H, refine=refine) is None


@pytest.mark.parametrize("refine", REFINE)
def test_directed_edge_direction(refine):
    # H: 0 -> 1; G: 0 -> 2 only. Must map onto the 0 -> 2 edge.
    H = [[0, 1], [0, 0]]
    G = [[0, 0, 1], [0, 0, 0], [0, 0, 0]]
    mapping = find_subgraph(G, H, refine=refine)
    assert mapping == {0: 0, 1: 2}
    assert verify_mapping(G, H, mapping)


@pytest.mark.parametrize("refine", REFINE)
def test_directed_no_match(refine):
    # H: 2-cycle 0 <-> 1; G: a directed path, no 2-cycle
    H = [[0, 1], [1, 0]]
    G = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    assert find_subgraph(G, H, refine=refine) is None


@pytest.mark.parametrize("refine", REFINE)
def test_self_loop_must_match(refine):
    H = [[1]]
    G = [[0, 1], [0, 0]]
    assert find_subgraph(G, H, refine=refine) is None

    G = [[0, 1], [0, 1]]
    mapping = find_subgraph(G, H, refine=refine)
    assert mapping == {0: 1}
    assert verify_mapping(G, H, mapping)


@pytest.mark.parametrize("refine", REFINE)
def test_no_self_loop_in_H(refine):
    # Induced: H vertex without a loop can't map to a G vertex with one
    H = [[0]]
    G = [[1]]
    assert find_subgraph(G, H, refine=refine) is None


@pytest.mark.parametrize("refine", REFINE)
def test_pattern_larger_than_target(refine):
    assert find_subgraph([[0]], [[0, 1], [1, 0]], refine=refine) is None


@pytest.mark.parametrize("refine", REFINE)
def test_path_in_triangle(refine):
    # A path is a non-induced subgraph of a triangle, but not an induced one
    path = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    triangle = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    assert find_subgraph(triangle, path, refine=refine) is None
    mapping = find_subgraph(triangle, path, induced=False, refine=refine)
    assert mapping is not None
    assert verify_mapping(triangle, path, mapping, induced=False)
    assert not verify_mapping(triangle, path, mapping)


@pytest.mark.parametrize("refine", REFINE)
def test_non_induced_self_loops(refine):
    # Non-induced: a loop in G is allowed where H has none, but not vice versa
    assert find_subgraph([[1]], [[0]], induced=False, refine=refine) == {0: 0}
    assert find_subgraph([[0]], [[1]], induced=False, refine=refine) is None


@pytest.mark.parametrize("refine", REFINE)
def test_non_induced_directed(refine):
    # Non-induced: a 2-cycle in G contains a single directed edge,
    # but a single directed edge does not contain a 2-cycle
    edge = [[0, 1], [0, 0]]
    cycle = [[0, 1], [1, 0]]
    assert find_subgraph(cycle, edge, induced=False, refine=refine) is not None
    assert find_subgraph(cycle, edge, refine=refine) is None
    assert find_subgraph(edge, cycle, induced=False, refine=refine) is None


def test_verify_mapping_injective():
    G = [[0, 1], [1, 0]]
    H = [[0, 0], [0, 0]]
    assert not verify_mapping(G, H, {0: 0, 1: 0}, induced=False)


@pytest.mark.parametrize("induced", [True, False])
@pytest.mark.parametrize("refine", REFINE)
def test_random_against_brute_force(refine, induced):
    rng = random.Random(0)
    for _ in range(300):
        directed = rng.random() < 0.5
        loops = rng.random() < 0.3
        n = rng.randint(1, 6)
        m = rng.randint(1, min(n, 4))
        G = random_graph(n, rng.random(), directed, loops, rng)
        H = random_graph(m, rng.random(), directed, loops, rng)
        mapping = find_subgraph(G, H, induced=induced, refine=refine)
        if mapping is None:
            assert not brute_force_exists(G, H, induced), (G, H)
        else:
            assert verify_mapping(G, H, mapping, induced), (G, H, mapping)


def test_invalid_refine():
    with pytest.raises(ValueError):
        find_subgraph([[0]], [[0]], refine="bogus")
