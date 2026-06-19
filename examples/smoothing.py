#!/usr/bin/env python3
"""
Smoothing

You're given a 2D grid of positive integers, where grid[i][j] represents the
brightness of a light fixture at position (i, j).

For safety/aesthetic reasons, any two adjacent fixtures (up, down, left,
right — no diagonals) can't differ in brightness by more than a factor of 2.
That is, for any pair of adjacent cells, the brighter one can be at most
double the dimmer one.

You're allowed to dim fixtures — decrease their brightness — but never
increase them above their original value. Given the input grid, return the
brightest possible grid that satisfies the adjacency constraint. In other
words, your output should be valid, and there should be no way to bump any
single cell's value back up (even by 1) without violating either "never
exceed the original" or the adjacency rule somewhere.

A small example to make sure the constraints are clear: for input [[1, 4]],
the output would be [[1, 2]] — fixture 0 stays at 1, and fixture 1 has to dim
from 4 down to 2, since it can be at most 2 × 1.

Take a moment — what's your approach? Feel free to ask clarifying questions
first (grid size bounds, are diagonals neighbors, etc.) before you start coding.

TODO: need to learn how to use heapq
"""


import numpy as np


class Grid():
    def __init__(self, grid):
        self.grid = np.array(grid)

    def run_dimming(self, threshold=2.0):
        pass


def main():
    grid = Grid([[1, 4]])
    grid.run_dimming()
    assert grid.grid.shape == (1, 2)
    assert np.array_equal(grid.grid, np.array([[1, 2]]))
    print("Pass!")


if __name__ == "__main__":
    main()
