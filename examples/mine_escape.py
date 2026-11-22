#!/usr/bin/env python3
"""
Mine escape (BFS)

A worker is in a mine, shown as a grid. S is the start, E is the exit, . is open ground and # is rock. Each move goes one cell up, down, left or right. Return the minimum number of moves to reach the exit, or -1 if it can't be reached.

grid = ["S..#....",
        ".#.#.##.",
        ".#...#..",
        ".####.#.",
        "......#E"]
# expected: 15

grid = ["S#.",
        "##.",
        "..E"]
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


def neighbors(grid, node):
    n_rows = len(grid)
    n_cols = len(grid[0])
    r, c = node
    nbrs = list()
    cross = [(r - 1, c), (r, c - 1), (r + 1, c), (r, c + 1)]
    for _r, _c in cross:
        if (0 <= _r < n_rows) and (0 <= _c < n_cols) and grid[_r][_c] != "#":
            nbrs.append((_r, _c))
    return nbrs


def bfs(grid, start, goal):
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
            return dist[node]
        # loop over neighbors
        for nbr in neighbors(grid, node):
            if nbr not in dist:
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    return -1


def mine_escape(grid):
    start = find_start(grid)
    goal = "E"
    search_result = bfs(grid, start, goal)
    return search_result


def main():
    grid = ["S..#....",
            ".#.#.##.",
            ".#...#..",
            ".####.#.",
            "......#E"]
    result = mine_escape(grid)
    print("Mine Escape: ", result)
    assert result == 15
    
    grid = ["S#.",
            "##.",
            "..E"]
    result = mine_escape(grid)
    print("Mine Escape: ", result)
    assert result == -1


if __name__ == "__main__":
    main()
