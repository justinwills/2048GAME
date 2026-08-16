from game2048 import create_board, add_random_tile, moves, can_move, max_tile

import math

TILE_CLASSES = [
    0, 2, 4, 8, 16, 32, 64, 128,
    256, 512, 1024, 2048,
    4096, 8192, 16384, 32768
]

TILE_TO_IDX = {v: i for i, v in enumerate(TILE_CLASSES)}


def encode_board(board):
    flat = [cell for row in board for cell in row]
    one_hot = [0.0] * (16 * len(TILE_CLASSES))
    for idx, value in enumerate(flat):
        one_hot[idx * len(TILE_CLASSES) + TILE_TO_IDX.get(value, 0)] = 1.0
    return one_hot

class Game2048Env:
    def __init__(self):
        self.board = None
        self.score = 0
        self.steps = 0

    def reset(self):
        self.board = create_board()
        add_random_tile(self.board)
        add_random_tile(self.board)
        self.score = 0
        self.steps = 0
        return encode_board(self.board)

    def get_state(self):
        # flatten the 4x4 board into 16 numbers
        return [cell for row in self.board for cell in row]
    
    def step(self, action):
        """
        action:
            0 = up
            1 = down
            2 = left
            3 = right
        """
        action_map = {
            0: "w",
            1: "s",
            2: "a",
            3: "d",
        }

        key = action_map[action]
        new_board, gained = moves[key](self.board)

        # illegal move: no change
        if new_board == self.board:
            reward = -20
            done = not can_move(self.board)
            return encode_board(self.board), reward, done, {}

        self.board = new_board
        self.score += gained
        self.steps += 1
        add_random_tile(self.board)

        done = not can_move(self.board)

        reward = gained

        empty_cells = sum(
            1 for row in self.board
            for cell in row
            if cell == 0
        )

        reward += empty_cells * 1.0
        
        monotonic_score = 0

        for row in self.board:
            if row[0] >= row[1] >= row[2] >= row[3]:
                monotonic_score += 1

        reward += monotonic_score * 10
        reward += 0.5

        tile = max_tile(self.board)
        
        corners = [
            self.board[0][0],
            self.board[0][3],
            self.board[3][0],
            self.board[3][3]
        ]

        if tile in corners:
            reward += 50
        else:
            reward -= 20
        
        if tile >= 256:
            reward += 20
        
        if tile >= 512:
            reward += 80

        if tile >= 1024:
            reward += 1000

        if tile >= 2048:
            reward += 7000

        milestone_bonus = 0
        if tile >= 512 and tile in corners:
            milestone_bonus = math.log2(tile) * 2   # 512→18, 1024→20, 2048→22
        elif tile >= 256 and tile in corners:
            milestone_bonus = math.log2(tile)

        reward += milestone_bonus

        # Monotonicity bonus: encourage tiles decreasing from one corner
        # (rewards board organization, which is prerequisite for 2048)
        def monotonicity_score(board):
            score = 0
            for row in board:
                for i in range(3):
                    if row[i] >= row[i+1]:
                        score += 1
            for c in range(4):
                col = [board[r][c] for r in range(4)]
                for i in range(3):
                    if col[i] >= col[i+1]:
                        score += 1
            return score / 24.0   # normalize to [0,1]

        reward += monotonicity_score(self.board) * 0.5

        if done:
            reward -= 100

        return encode_board(self.board), reward, done, {}

    def render(self):
        for row in self.board:
            print(row)
        print("score:", self.score, "steps:", self.steps)
        
    def valid_actions(self):
        valid = []

        action_map = {
        0: "w",
        1: "s",
        2: "a",
        3: "d"
        }

        for action, key in action_map.items():
            new_board, _ = moves[key](self.board)

            if new_board != self.board:
                valid.append(action)

        return valid