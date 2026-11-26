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

---

Stage 2: food and growth

Now the snake has a body, and there's food.

- food is a list of cells, [(r, c), ...]. Only one food item is on the board at a time: first food[0]; once it's eaten, food[1] appears; and so on.
- When the head moves onto the current food, the score goes up by 1 and the snake grows by one cell: the head moves forward and the tail stays where it was.
- Otherwise the snake moves normally: the head moves forward and the tail end moves up by one cell.
- The game also ends if the head runs into the snake's own body.

Write play(H, W, food, moves) that returns a tuple (score, moves_completed).

play(3, 3, [(0,1), (0,2), (1,2)], "RRDLLU")           # -> (3, 6)
play(3, 3, [(0,1), (0,2), (1,2), (1,1)], "RRDLU")     # -> (4, 4)  the last U hits its own body
play(2, 2, [(0,1), (1,1), (1,0)], "RDLU")             # -> (3, 4)
"""

from collections import deque


def is_valid_pos(n_rows, n_cols, snake_set, pos):
    _r, _c = pos
    return (0 <= _r < n_rows) and (0 <= _c < n_cols) and (pos not in snake_set)


def validate_move(n_rows, n_cols, snake, snake_set, food, move):
    assert len(snake) > 0
    pos = snake[-1]
    points = 0

    _r, _c = pos
    if move == "U":
        _r -= 1
    elif move == "D":
        _r += 1
    elif move == "L":
        _c -= 1
    elif move == "R":
        _c += 1
    else:
        assert False

    pos = (_r, _c)

    if food and (food[0] == pos):
        _ = food.popleft()
        points += 1
    else:
        _pos = snake.popleft()
        snake_set.remove(_pos)

    if is_valid_pos(n_rows, n_cols, snake_set, pos):
        return (points, pos)
    else:
        return (0, None)


def play(H, W, food, moves):
    """
    Returns a tuple (score, moves_completed).

    play(3, 3, [(0,1), (0,2), (1,2)], "RRDLLU")           # -> (3, 6)
    play(3, 3, [(0,1), (0,2), (1,2), (1,1)], "RRDLU")     # -> (4, 4)  the last U hits its own body
    play(2, 2, [(0,1), (1,1), (1,0)], "RDLU")             # -> (3, 4)
    """
    pos = (0, 0)  # start
    n_moves = 0
    score = 0
    snake = deque([pos])
    snake_set = set([pos])
    food = deque(food)

    for move in moves:
        points, pos = validate_move(H, W, snake, snake_set, food, move)
        if pos is None:
            break
        snake.append(pos)
        snake_set.add(pos)
        n_moves += 1
        score += points
    return (score, n_moves)


def main():
    result = play(3, 3, [(0,1), (0,2), (1,2)], "RRDLLU")           # -> (3, 6)
    print(result)
    assert result == (3, 6)
    result = play(3, 3, [(0,1), (0,2), (1,2), (1,1)], "RRDLU")     # -> (4, 4)  the last U hits its own body
    print(result)
    assert result == (4, 4)
    result = play(2, 2, [(0,1), (1,1), (1,0)], "RDLU")             # -> (3, 4)
    print(result)
    assert result == (3, 4)


if __name__ == "__main__":
    main()
