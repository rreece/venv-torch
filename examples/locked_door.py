#!/usr/bin/env python3
"""
Problem 3: Locked door (BFS where the state is more than a position)

This is the same as Problem 1 with two additions: k is a key and D is a door. You can walk through D only after you've picked up the key, which happens when you step on k. Return the minimum number of moves from S to E, or -1.

grid = ["S.#k",
        "..#.",
        "....",
        "##D#",
        "...E"]
# expected: 13

grid = ["S..D.E",
        "#####.",
        "k....."]
# expected: -1
"""

from collections import deque


def find_start(grid):
    start = None
    n_rows = len(grid)
    n_cols = len(grid[0])
    for r in range(n_rows):
        assert n_cols == len(grid[r])
        for c in range(n_cols):
            if grid[r][c] == "S":
                start = (r, c)
                return start
    return start


def get_neighbors(grid, node, obstacles):
    n_rows = len(grid)
    n_cols = len(grid[0])
    r, c = node
    nbrs = list()
    cross = [(r-1, c), (r, c+1), (r+1, c), (r, c-1)]
    for _r, _c in cross:
        if (0 <= _r < n_rows) and (0 <= _c < n_cols) and grid[_r][_c] not in obstacles:
            nbrs.append((_r, _c))
    return nbrs


def bfs(grid, start, goal, obstacles):
    # start queue
    queue = deque([start])
    dist = dict()
    dist[start] = 0
    # while queue
    while queue:
        # pop from queue
        node = queue.popleft()
        _r, _c = node
        # check for goal
        if grid[_r][_c] == goal:
            return dist[node], node
        # loop over neighbors
        for nbr in get_neighbors(grid, node, obstacles):
            if nbr not in dist:
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    return -1, None


def mine_escape(grid):
    start = find_start(grid)
    results = list()

    # Route A: first, try to find the exit (without the key)
    goal = "E"
    obstacles = set(["#", "D"])
    dist, node = bfs(grid, start, goal, obstacles)
    results.append(dist)

    # Route B: find the key
    goal = "k"
    obstacles = set(["#", "D"])
    dist, node = bfs(grid, start, goal, obstacles)
    result = dist

    if result >= 0:
        # find the exit
        goal = "E"
        obstacles = set(["#"])
        dist, node = bfs(grid, node, goal, obstacles)
        if dist >= 0:
            result += dist
        else:
            result = -1

    results.append(result)

    results = list(filter(lambda x: x >= 0, results))
    results.sort()

    if len(results) >= 1:
        return results[0]
    else:
        return -1


def main():
    grid = ["S.#k",
            "..#.",
            "....",
            "##D#",
            "...E"]
    # expected: 13
    result = mine_escape(grid)
    print("Locked Door: ", result)
    assert result == 13
    
    grid = ["S..D.E",
            "#####.",
            "k....."]
    # expected: -1
    result = mine_escape(grid)
    print("Locked Door: ", result)
    assert result == -1

    grid = ["k.S",
            "##D",
            "..E"]
    # expected: 6 (key is left of S)
    result = mine_escape(grid)
    print("Locked Door: ", result)
    assert result == 6

    grid = ["S.E"]
    # expected: 2 (no key, no door needed)
    result = mine_escape(grid)
    print("Locked Door: ", result)
    assert result == 2

    grid = ["S.E",
            "...",
            "k.."]
    # expected: 2 (key exists but going straight is shorter)
    result = mine_escape(grid)
    print("Locked Door: ", result)
    assert result == 2

    grid = ["Sk#E"]
    # expected: -1 (key is reachable, but the exit is not reachable from the key)
    result = mine_escape(grid)
    print("Locked Door: ", result)
    assert result == -1


if __name__ == "__main__":
    main()
