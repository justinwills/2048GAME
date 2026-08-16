import math
import random

SIZE = 4

weights = {
    "empty": 2200,
    "biggest": 3,
    "smooth": 0.3,
    "corner": 8,
    "mono": 3
}

def log2_tile(value):
    return math.log2(value) if value > 0 else 0

def create_board():
    return [[0 for _ in range(SIZE)] for _ in range(SIZE)]


def add_random_tile(board):
    empty_cells = []

    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] == 0:
                empty_cells.append((r, c))

    if not empty_cells:
        return board

    r, c = random.choice(empty_cells)
    board[r][c] = 2 if random.random() < 0.9 else 4
    return board


def print_board(board, score):
    print("Score:", score)
    for row in board:
        print(row)
    print()


def move_row_left(row):
    new_row = [x for x in row if x != 0]

    result = []
    score = 0
    i = 0

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
    new_board = []
    total_score = 0

    for row in board:
        new_row, score = move_row_left(row)
        new_board.append(new_row)
        total_score += score

    return new_board, total_score


def reverse_board(board):
    return [row[::-1] for row in board]


def transpose(board):
    return [list(row) for row in zip(*board)]


def move_right(board):
    reversed_board = reverse_board(board)
    moved_board, score = move_left(reversed_board)
    final_board = reverse_board(moved_board)
    return final_board, score


def move_up(board):
    transposed = transpose(board)
    moved_board, score = move_left(transposed)
    final_board = transpose(moved_board)
    return final_board, score


def move_down(board):
    transposed = transpose(board)
    moved_board, score = move_right(transposed)
    final_board = transpose(moved_board)
    return final_board, score


moves = {
    "w": move_up,
    "s": move_down,
    "a": move_left,
    "d": move_right
}


def can_move(board):
    if any(0 in row for row in board):
        return True

    for move_function in moves.values():
        new_board, _ = move_function(board)
        if new_board != board:
            return True

    return False


def count_empty(board):
    count = 0
    for row in board:
        count += row.count(0)
    return count


def max_tile(board):
    return max(max(row) for row in board)


def smoothness(board):
    penalty = 0

    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] == 0:
                continue

            current = log2_tile(board[r][c])

            if c + 1 < SIZE and board[r][c + 1] != 0:
                neighbor = log2_tile(board[r][c + 1])
                penalty += abs(current - neighbor)

            if r + 1 < SIZE and board[r + 1][c] != 0:
                neighbor = log2_tile(board[r + 1][c])
                penalty += abs(current - neighbor)

    return -penalty


def corner_bonus(board):
    biggest = max_tile(board)

    if biggest in [board[0][0], board[0][3], board[3][0], board[3][3]]:
        return 1

    return 0


def corner_locked_bonus(board):
    biggest = max_tile(board)

    if biggest in [board[0][0], board[0][3], board[3][0], board[3][3]]:
        return biggest * 30

    return 0


def monotonicity(board):
    total = 0

    lines = []
    for row in board:
        lines.append([log2_tile(v) for v in row])

    for c in range(SIZE):
        lines.append([log2_tile(board[r][c]) for r in range(SIZE)])

    for line in lines:
        inc = 0
        dec = 0

        for i in range(SIZE - 1):
            if line[i] >= line[i + 1]:
                dec += line[i] - line[i + 1]
            else:
                inc += line[i + 1] - line[i]

        total += max(inc, dec)

    return total


def evaluate(board, weights):
    empty = count_empty(board)
    biggest = max_tile(board)

    smooth = smoothness(board)
    corner = corner_bonus(board)
    mono = monotonicity(board)

    high_tile_bonus = 0

    if biggest >= 1024:
        high_tile_bonus += biggest * 50

    if biggest >= 2048:
        high_tile_bonus += biggest * 200

    score = (
        empty * weights["empty"]
        + biggest * weights["biggest"]
        + smooth * weights["smooth"]
        + corner * weights["corner"]
        + mono * weights["mono"]
        + high_tile_bonus
    )

    return score
 
def search_score(board, depth, weights):
    if depth == 0 or not can_move(board):
        return evaluate(board, weights)

    best_score = -999999999

    for key, move_function in moves.items():
        new_board, gained = move_function(board)

        if new_board == board:
            continue

        score = gained + search_score(new_board, depth - 1, weights)

        if score > best_score:
            best_score = score

    return best_score


def get_empty_cells(board):
    cells = []

    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] == 0:
                cells.append((r, c))

    return cells


def copy_board(board):
    return [row[:] for row in board]


def expectimax(board, depth, is_player_turn):
    if depth == 0 or not can_move(board):
        return evaluate(board, weights)

    if is_player_turn:
        best_score = -999999999

        for key, move_function in moves.items():
            new_board, gained = move_function(board)

            if new_board == board:
                continue

            score = gained + expectimax(new_board, depth - 1, False)

            if score > best_score:
                best_score = score

        return best_score

    else:
        empty_cells = get_empty_cells(board)

        if not empty_cells:
            return evaluate(board, weights)

        total_score = 0

        for r, c in empty_cells:
            board_2 = copy_board(board)
            board_2[r][c] = 2
            total_score += 0.9 * expectimax(board_2, depth - 1, True)

            board_4 = copy_board(board)
            board_4[r][c] = 4
            total_score += 0.1 * expectimax(board_4, depth - 1, True)

        return total_score / len(empty_cells)


def ai_move(board, depth=3):
    best_score = -999999999
    best_move = None

    for key, move_function in moves.items():
        new_board, gained = move_function(board)

        if new_board == board:
            continue

        score = gained + expectimax(new_board, depth - 1, False)

        if score > best_score:
            best_score = score
            best_move = key

    return best_move

def choose_depth(board):
    biggest = max_tile(board)

    if biggest >= 1024:
        return 5
    elif biggest >= 512:
        return 4
    else:
        return 3
    
def play_game(show=False, depth=4):
    board = create_board()
    add_random_tile(board)
    add_random_tile(board)

    score = 0
    steps = 0

    while can_move(board):
        if show:
            print_board(board, score)

        current_depth = choose_depth(board)

        command = ai_move(board, depth=current_depth)

        if command is None:
            break

        new_board, gained = moves[command](board)

        if new_board != board:
            board = new_board
            score += gained
            steps += 1
            add_random_tile(board)

    return score, max_tile(board), steps


def run_experiments(n=200, depth=4):
    scores = []
    max_tiles = []
    steps_list = []
    tile_counts = {}

    for i in range(n):
        score, tile, steps = play_game(show=False, depth=depth)

        scores.append(score)
        max_tiles.append(tile)
        steps_list.append(steps)

        if tile not in tile_counts:
            tile_counts[tile] = 0

        tile_counts[tile] += 1

        print(f"Game {i + 1}: Score={score}, MaxTile={tile}, Steps={steps}")

    print("\n===== RESULTS =====")
    print("Games:", n)
    print("Depth:", depth)

    print("\nAverage Score:", sum(scores) / n)
    print("Best Score:", max(scores))

    print("\nAverage Max Tile:", sum(max_tiles) / n)
    print("Best Max Tile:", max(max_tiles))

    print("\nAverage Steps:", sum(steps_list) / n)

    print("\n===== TILE DISTRIBUTION =====")

    for tile in sorted(tile_counts.keys()):
        count = tile_counts[tile]
        percentage = (count / n) * 100
        print(f"{tile}: {count} games ({percentage:.2f}%)")


if __name__ == "__main__":
    run_experiments(n=20, depth=4)