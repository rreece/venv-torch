#!/usr/bin/env python3
"""
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
"""

from collections import deque


def get_neighbors(grid, node):
    n_rows = len(grid)
    n_cols = len(grid[0])
    r, c = node
    nbrs = list()
    cross = [(r-1, c), (r-1, c+1), (r, c+1), (r+1, c+1), (r+1, c), (r+1, c-1), (r, c-1), (r-1, c-1)]
    for _r, _c in cross:
        if (0 <= _r < n_rows) and (0 <= _c < n_cols):
            nbrs.append((_r, _c))
    return nbrs


def bfs(grid, start):
    # start queue
    queue = deque([start])
    visited = set()
    visited.add(start)
    order = list()
    # while queue
    while queue:
        # pop from queue
        node = queue.popleft()
        order.append(node)
        # loop over neighbors
        neighbors = get_neighbors(grid, node)
        n_mines = 0
        for nbr in neighbors:
            _r, _c = nbr
            if grid[_r][_c] == "*":
                n_mines += 1
        if n_mines == 0:
            for nbr in neighbors:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
    return order


def minesweeper(board, click):
    """
    1. The goal: Problem 1 searched for one cell. What are you computing here, and when does the search stop?
    2. Neighbors: there are 8 directions instead of 4. You'll also need to count adjacent mines for a cell. Can one helper do both jobs, or do you need two?
    3. The spreading rule: every safe neighbor is revealed, but only some revealed cells keep spreading. Which ones? This is the heart of the problem.
    4. The container: does order matter here, or would a stack, queue or recursion all give the same answer? (This is the question from earlier.)
    5. Edge case: clicking a mine.
    """
    _r, _c = click
    if board[_r][_c] == "*":
        # mine
        return -1

    field = bfs(board, click)
    return len(field)


def main():
    board = [".....",
             ".....",
             "..*..",
             ".....",
             "*...."]
    # click (0, 0) -> 23
    result = minesweeper(board, click=(0, 0))
    print("minesweeper(board, click=(0, 0)) = ", result)
    assert result == 23
    # click (3, 1) -> 1     (it touches a mine, so nothing spreads)
    result = minesweeper(board, click=(3, 1))
    print("minesweeper(board, click=(3, 1)) = ", result)
    assert result == 1
    # click (2, 2) -> -1
    result = minesweeper(board, click=(2, 2))
    print("minesweeper(board, click=(2, 2)) = ", result)
    assert result == -1


if __name__ == "__main__":
    main()
