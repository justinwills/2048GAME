import csv

from game2048 import (
    create_board,
    add_random_tile,
    can_move,
    moves,
    ai_move
)

MOVE_TO_LABEL = {
    "w": 0,
    "s": 1,
    "a": 2,
    "d": 3
}

with open("dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)

    for game in range(5000):

        board = create_board()
        add_random_tile(board)
        add_random_tile(board)

        while can_move(board):

            move = ai_move(board)

            if move is None:
                break

            state = [
                cell
                for row in board
                for cell in row
            ]

            writer.writerow(
                state +
                [MOVE_TO_LABEL[move]]
            )

            board, _ = moves[move](board)

            add_random_tile(board)

print("Dataset created!")