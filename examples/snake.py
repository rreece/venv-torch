#!/usr/bin/env python3
"""
Problem: Snake

We're building the logic for a simple Snake game. The board is a grid with H rows and W columns. Row 0 is at the top, column 0 at the left.

Stage 1: the snake is a single cell that starts at (0, 0). You're given a string of moves, each one of U, D, L or R (up, down, left, right). Apply them in order. If a move would take the snake off
the board, the game ends immediately.

Write play(H, W, moves) that returns the number of moves completed successfully before the game ended, or len(moves) if it never ended.

play(3, 4, "RRDRU")   # -> 5   (never leaves the board)
play(3, 4, "RRRR")    # -> 3   (the 4th R would move to column 4, off the board)
play(3, 4, "DDD")     # -> 2
"""


def is_valid_pos(n_rows, n_cols, pos):
    _r, _c = pos
    return (0 <= _r < n_rows) and (0 <= _c < n_cols)


def validate_move(n_rows, n_cols, pos, move):
    if pos is None:
        return None

    _r, _c = pos
    if move == "U":
        _r -= 1
    elif move == "D":
        _r += 1
    elif move == "L":
        _c -= 1
    elif move == "R":
        _c += 1

    pos = (_r, _c)
    if is_valid_pos(n_rows, n_cols, pos):
        return pos
    else:
        return None


def play(H, W, moves):
    """
    Returns the number of moves completed successfully before the game ended, or len(moves) if it never ended.

    play(3, 4, "RRDRU")   # -> 5   (never leaves the board)
    play(3, 4, "RRRR")    # -> 3   (the 4th R would move to column 4, off the board)
    play(3, 4, "DDD")     # -> 2
    """
    pos = (0, 0)  # start
    n_moves = 0
    for move in moves:
        pos = validate_move(H, W, pos, move)
        if pos is None:
            break
        n_moves += 1
    return n_moves 


def main():
    result = play(3, 4, "RRDRU")   # -> 5   (never leaves the board)
    assert result == 5
    result = play(3, 4, "RRRR")    # -> 3   (the 4th R would move to column 4, off the board)
    assert result == 3
    result = play(3, 4, "DDD")     # -> 2
    assert result == 2


if __name__ == "__main__":
    main()
