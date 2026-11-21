#!/usr/bin/env python3
"""
Examples of breadth-first search (BFS) and depth-first search (DFS).

---

Problem 1: Mine escape (BFS)

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

---

Problem 2: Minesweeper click (DFS / flood fill)

A board has * for mines and . for safe cells. The player clicks cell (r, c):
- If it's a mine, return -1.
- Otherwise the cell is revealed. If none of its 8 neighbors is a mine, all of its neighbors are revealed automatically, and this repeats for any revealed neighbor that also has no adjacent mines.

Return the number of cells revealed.

board = [".....",
         ".....",
         "..*..",
         ".....",
         "*...."]
# click (0, 0) -> 23
# click (3, 1) -> 1     (it touches a mine, so nothing spreads)
# click (2, 2) -> -1

---

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
    visited = set()
    queue = deque([start])
    visited.add(start)
    dist = dict()
    dist[start] = 0
    while queue:
        node = queue.popleft()
        _r, _c = node
        if grid[_r][_c] == goal:
            return dist[node]
        for nbr in neighbors(grid, node):
            if not nbr in visited:
                visited.add(nbr)
                queue.append(nbr)
                dist[nbr] = dist[node] + 1
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
    # expected: 15
    
    grid = ["S#.",
            "##.",
            "..E"]
    # expected: -1
    result = mine_escape(grid)
    print("Mine Escape: ", result)



if __name__ == "__main__":
    main()
