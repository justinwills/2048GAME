"""
game_engine.py — 2048 core game logic (importable module)
==========================================================
Extracted from 2048.py so that evolve.py, neural net training, and RL
modules can import game primitives without triggering run_experiments().

Public API
----------
create_board()
add_random_tile(board)
can_move(board)
moves            : dict[str -> move_function]
evaluate(board, weights)
ai_move_with_weights(board, weights, depth)
play_game_with_weights(weights, depth, show) -> (score, max_tile, steps)
play_game(show, depth)          # uses default weights
count_empty(board)
max_tile(board)
"""

import random

SIZE = 4

# ── default weights (hand-tuned baseline) ───────────────────────────────────
BASELINE_WEIGHTS = {
    "empty": 2200,
    "biggest": 3,
    "smooth": 0.3,
    "corner": 8,
    "mono": 3,
}

DEFAULT_WEIGHTS = {
    "empty": 1284.8,
    "biggest": 0.5,
    "smooth": 0.9,
    "corner": 26.2,
    "mono": 20.0,
}


# ── board primitives ─────────────────────────────────────────────────────────

def create_board():
    return [[0] * SIZE for _ in range(SIZE)]


def add_random_tile(board):
    empty = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == 0]
    if not empty:
        return board
    r, c = random.choice(empty)
    board[r][c] = 2 if random.random() < 0.9 else 4
    return board


def copy_board(board):
    return [row[:] for row in board]


def get_empty_cells(board):
    return [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == 0]


def count_empty(board):
    return sum(row.count(0) for row in board)


def max_tile(board):
    return max(max(row) for row in board)


# ── move logic ───────────────────────────────────────────────────────────────

def move_row_left(row):
    new_row = [x for x in row if x != 0]
    result, score, i = [], 0, 0
    while i < len(new_row):
        if i + 1 < len(new_row) and new_row[i] == new_row[i + 1]:
            merged = new_row[i] * 2
            result.append(merged)
            score += merged
            i += 2
        else:
            result.append(new_row[i])
            i += 1
    while len(result) < SIZE:
        result.append(0)
    return result, score


def move_left(board):
    new_board, total = [], 0
    for row in board:
        new_row, s = move_row_left(row)
        new_board.append(new_row)
        total += s
    return new_board, total


def reverse_board(board):
    return [row[::-1] for row in board]


def transpose(board):
    return [list(row) for row in zip(*board)]


def move_right(board):
    rb, s = move_left(reverse_board(board))
    return reverse_board(rb), s


def move_up(board):
    tb, s = move_left(transpose(board))
    return transpose(tb), s


def move_down(board):
    tb, s = move_right(transpose(board))
    return transpose(tb), s


moves = {"w": move_up, "s": move_down, "a": move_left, "d": move_right}


def can_move(board):
    if any(0 in row for row in board):
        return True
    for fn in moves.values():
        nb, _ = fn(board)
        if nb != board:
            return True
    return False


# ── heuristic evaluation ─────────────────────────────────────────────────────

def smoothness(board):
    penalty = 0
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] == 0:
                continue
            if c + 1 < SIZE and board[r][c + 1] != 0:
                penalty += abs(board[r][c] - board[r][c + 1])
            if r + 1 < SIZE and board[r + 1][c] != 0:
                penalty += abs(board[r][c] - board[r + 1][c])
    return -penalty


def corner_bonus(board):
    biggest = max_tile(board)
    if board[0][0] == biggest:
        return biggest * 20
    return 0


def corner_locked_bonus(board):
    biggest = max_tile(board)
    if biggest in [board[0][0], board[0][3], board[3][0], board[3][3]]:
        return biggest * 30
    return 0


def monotonicity(board):
    score = 0
    for row in board:
        for i in range(SIZE - 1):
            if row[i] >= row[i + 1]:
                score += row[i]
    for c in range(SIZE):
        for r in range(SIZE - 1):
            if board[r][c] >= board[r + 1][c]:
                score += board[r][c]
    return score


def evaluate(board, weights):
    empty   = count_empty(board)
    biggest = max_tile(board)
    smooth  = smoothness(board)
    corner  = corner_bonus(board)
    locked  = corner_locked_bonus(board)
    mono    = monotonicity(board)

    high_tile_bonus = 0
    if biggest >= 1024:
        high_tile_bonus += biggest * 50
    if biggest >= 2048:
        high_tile_bonus += biggest * 200

    return (
        empty   * weights["empty"]
        + biggest * weights["biggest"]
        + smooth  * weights["smooth"]
        + corner  * weights["corner"]
        + locked
        + mono    * weights["mono"]
        + high_tile_bonus
    )


# ── expectimax search ────────────────────────────────────────────────────────

def expectimax(board, depth, is_player_turn, weights):
    if depth == 0 or not can_move(board):
        return evaluate(board, weights)

    if is_player_turn:
        best = -1e18
        for fn in moves.values():
            nb, gained = fn(board)
            if nb == board:
                continue
            score = gained + expectimax(nb, depth - 1, False, weights)
            if score > best:
                best = score
        return best
    else:
        empty_cells = get_empty_cells(board)
        if not empty_cells:
            return evaluate(board, weights)
        total = 0
        for r, c in empty_cells:
            b2 = copy_board(board); b2[r][c] = 2
            b4 = copy_board(board); b4[r][c] = 4
            total += 0.9 * expectimax(b2, depth - 1, True, weights)
            total += 0.1 * expectimax(b4, depth - 1, True, weights)
        return total / len(empty_cells)


def ai_move_with_weights(board, weights, depth=3):
    best_score, best_move = -1e18, None
    for key, fn in moves.items():
        nb, gained = fn(board)
        if nb == board:
            continue
        score = gained + expectimax(nb, depth - 1, False, weights)
        if score > best_score:
            best_score, best_move = score, key
    return best_move


def choose_depth(board):
    t = max_tile(board)
    if t >= 1024: return 5
    if t >= 512:  return 4
    return 3


# ── game runner ──────────────────────────────────────────────────────────────

def play_game_with_weights(weights, depth=4, show=False):
    board = create_board()
    add_random_tile(board)
    add_random_tile(board)
    score, steps = 0, 0

    while can_move(board):
        if show:
            print(f"Score: {score}")
            for row in board: print(row)
            print()

        d    = choose_depth(board) if depth == 0 else depth
        cmd  = ai_move_with_weights(board, weights, depth=d)
        if cmd is None:
            break

        nb, gained = moves[cmd](board)
        if nb != board:
            board  = nb
            score += gained
            steps += 1
            add_random_tile(board)

    return score, max_tile(board), steps


def play_game(show=False, depth=4):
    return play_game_with_weights(DEFAULT_WEIGHTS, depth=depth, show=show)
