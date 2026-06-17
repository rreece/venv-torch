#!/usr/bin/env python3
"""
Examples of breadth-first search (BFS) and depth-first search (DFS).

Tree used in examples:

        A
       / \\
      B   C
     / \\   \\
    D   E   F

Adjacency list (undirected):
    A: B, C
    B: A, D, E
    C: A, F
    D: B
    E: B
    F: C
"""


from collections import deque


GRAPH = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B"],
    "F": ["C"],
}


# --- Breadth-First Search ---
# Uses a queue (deque). Explores all neighbors at the current depth
# before moving deeper. Finds the shortest path in an unweighted graph.

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


# --- Depth-First Search (recursive) ---
# Explores as deep as possible before backtracking.
# visited is passed through to avoid revisiting nodes.

def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    order = [node]

    for neighbor in graph[node]:
        if neighbor not in visited:
            order += dfs_recursive(graph, neighbor, visited)

    return order


# --- Depth-First Search (iterative) ---
# Same logic as BFS but uses a stack (list) with pop() instead of
# a queue with popleft(). LIFO order produces depth-first traversal.

def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    order = []

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                stack.append(neighbor)

    return order


def main():
    print("BFS from A:           ", bfs(GRAPH, "A"))
    print("DFS recursive from A: ", dfs_recursive(GRAPH, "A"))
    print("DFS iterative from A: ", dfs_iterative(GRAPH, "A"))


if __name__ == "__main__":
    main()
