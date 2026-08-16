import os
import heapq
import torch
import torch.nn as nn
import numpy as np

from collections import deque
import random
import copy
import math
from env_2048 import Game2048Env

class MoveNet(nn.Module):
    """
    3-layer MLP for move classification.

    Architecture
    ------------
    Input (256)
    → Linear(256 → 512) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(512 → 256) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.2)
    → Linear(128 →   4) → (softmax at inference)
    """
    def __init__(self, input_dim=256, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

TILE_CLASSES = [
    0, 2, 4, 8, 16, 32, 64, 128,
    256, 512, 1024, 2048,
    4096, 8192, 16384, 32768
]

TILE_TO_IDX = {
    v: i for i, v in enumerate(TILE_CLASSES)
}

def encode_board(board):
    vec = np.zeros((4, 4, len(TILE_CLASSES)), dtype=np.float32)

    for r in range(4):
        for c in range(4):
            value = board[r][c]
            idx = TILE_TO_IDX.get(value, 0)
            vec[r, c, idx] = 1.0

    return vec.flatten()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

base_result_dir = os.path.join(ROOT_DIR, "result")
i = 1
while True:
    RESULT_DIR = f"{base_result_dir}_{i}"
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        break
    i += 1

print(f"Results will be saved in: {RESULT_DIR}")

model = MoveNet()

BEST_AVG_MODEL = os.path.join(ROOT_DIR, "result", "best_avg_model.pth")

if os.path.exists(BEST_AVG_MODEL):
    model.load_state_dict(
        torch.load(BEST_AVG_MODEL, map_location="cpu")
    )
    print("Loaded best_avg_model.pth")
else:
    model.load_state_dict(
        torch.load(
            os.path.join(ROOT_DIR, "models", "sl_model.pt"),
            map_location="cpu"
        )
    )
    print("Loaded sl_model.pt")

target_model = MoveNet()
target_model.load_state_dict(
    model.state_dict()
)

teacher_model = MoveNet()
teacher_model.load_state_dict(
    torch.load(os.path.join(ROOT_DIR, "models", "sl_model.pt"),
                map_location="cpu")
)
teacher_model.eval()

for p in teacher_model.parameters():
    p.requires_grad = False

memory_capacity = 200000
memory = deque(maxlen=memory_capacity)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2e-6
)

# Huber loss for more stable Q-learning
loss_fn = nn.SmoothL1Loss()

gamma = 0.99
epsilon_start = 0.01
epsilon_end = 0.001
epsilon_decay = 30000
teacher_weight_start = 1.0
teacher_weight_end = 0.8
teacher_decay_steps = 200000
train_freq = 32
train_steps_per_update = 1
batch_size = 128
target_update_steps = 2000
frame_idx = 0

model.eval()

print("Model loaded successfully!")

def board_heuristic(board):
    empty_cells = sum(
        1 for row in board
        for cell in row
        if cell == 0
    )

    max_tile = max(
        max(row)
        for row in board
    )

    corners = [
        board[0][0],
        board[0][3],
        board[3][0],
        board[3][3]
    ]

    score = 0

    # Empty cells are very important for 2048
    score += empty_cells * 120
    
    if empty_cells <= 2:
        score -= 3000

    # Keep biggest tile in a corner
    if max_tile in corners:
        score += 3000
    else:
        score -= 500

    # Reward monotonic rows
    for row in board:
        if row[0] >= row[1] >= row[2] >= row[3]:
            score += 200
        if row[3] >= row[2] >= row[1] >= row[0]:
            score += 200

    # Reward monotonic columns
    for c in range(4):
        col = [
            board[0][c],
            board[1][c],
            board[2][c],
            board[3][c]
        ]

        if col[0] >= col[1] >= col[2] >= col[3]:
            score += 200
        if col[3] >= col[2] >= col[1] >= col[0]:
            score += 200

    # Reward possible merges
    for r in range(4):
        for c in range(3):
            if board[r][c] != 0 and board[r][c] == board[r][c + 1]:
                score += math.log2(board[r][c]) * 80

    for c in range(4):
        for r in range(3):
            if board[r][c] != 0 and board[r][c] == board[r + 1][c]:
                score += math.log2(board[r][c]) * 80

    # Strong reward for high tile
    if max_tile >= 512:
        score += 500

    if max_tile >= 1024:
        score += 5000

    if max_tile >= 2048:
        score += 10000

    return score


def choose_action(env, logits, epsilon):
    valid_actions = env.valid_actions()

    q_values = logits.squeeze().detach().numpy()

    if random.random() < epsilon:
        return random.choice(valid_actions)

    best_action = None
    best_value = -999999999

    for action in valid_actions:
        sim_env = copy.deepcopy(env)

        try:
            sim_env.step(action)
            heuristic_value = board_heuristic(sim_env.board)
        except:
            heuristic_value = -999999

        combined_value = q_values[action] + 0.07 * heuristic_value

        if combined_value > best_value:
            best_value = combined_value
            best_action = action

    return best_action


def train_step(batch_size=batch_size):
    if len(memory) < batch_size:
        return

    model.train()

    batch = random.sample(
        memory,
        batch_size
    )

    states = torch.FloatTensor(np.array([x[0] for x in batch]))
    actions = torch.LongTensor([x[1] for x in batch]).unsqueeze(1)
    rewards = torch.FloatTensor([x[2] for x in batch]).unsqueeze(1)
    next_states = torch.FloatTensor(np.array([x[3] for x in batch]))
    dones = torch.FloatTensor([x[4] for x in batch]).unsqueeze(1)

    current_q = model(states).gather(1, actions)

    # Double DQN
    with torch.no_grad():
        next_actions = model(next_states).argmax(dim=1, keepdim=True)
        next_q = target_model(next_states).gather(1, next_actions).squeeze(1).unsqueeze(1)
        target_q = rewards + gamma * next_q * (1 - dones)

    rl_loss = loss_fn(current_q, target_q)

    # teacher supervision (small auxiliary loss)
    with torch.no_grad():
        teacher_logits = teacher_model(states)

    student_logits = model(states)
    teacher_actions = teacher_logits.argmax(dim=1)
    teacher_loss = nn.CrossEntropyLoss()(student_logits, teacher_actions)

    current_teacher_weight = max(
        teacher_weight_end,
        teacher_weight_start - frame_idx * (teacher_weight_start - teacher_weight_end) / teacher_decay_steps
    )

    loss = 0.02 * rl_loss + current_teacher_weight * teacher_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    model.eval()
    
def play_one_game():
    global frame_idx

    env = Game2048Env()
    env.reset()

    done = False
    
    game_memory = []
    
    steps = 0

    while not done:

        board_features = encode_board(env.board)

        x = torch.FloatTensor(
            board_features
        ).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)

        current_epsilon = max(
            epsilon_end,
            epsilon_start - frame_idx * (
                epsilon_start - epsilon_end
            ) / epsilon_decay
        )

        action = choose_action(
            env,
            logits,
            current_epsilon
        )

        old_state = board_features.copy()

        next_state, reward, done, _ = env.step(action)
        
        steps += 1

        new_state = encode_board(
            env.board
        )

        game_memory.append(
            (
                old_state,
                action,
                reward,
                new_state,
                done
            )
        )

        frame_idx += 1

        if len(memory) >= 3000 and frame_idx % train_freq == 0:
            for _ in range(train_steps_per_update):
                train_step()

        if frame_idx % target_update_steps == 0:
            target_model.load_state_dict(
                model.state_dict()
            )

    max_tile_value = max(
        max(row)
        for row in env.board
    )
    
    if env.score >= 4500 or max_tile_value >= 512:
        memory.extend(game_memory)
    else:
        memory.extend(game_memory[-30:])

    if max_tile_value >= 1024 and env.score >= 13000:

        model_name = f"model_1024_score_{env.score}.pth"
        model_path = os.path.join(RESULT_DIR, model_name)

        print(model_name)

        torch.save(
            model.state_dict(),
            model_path
        )

    if max_tile_value >= 2048:
    
        model_name = f"model_2048_score_{env.score}.pth"
        model_path = os.path.join(RESULT_DIR, model_name)

        print("\n")
        print("###################")
        print(model_name)
        print("###################")
        print("\n")

        torch.save(
            model.state_dict(),
            model_path
        )

    return env.score, max_tile_value, steps
        
scores = []
tiles = []
steps_list = []
tile_counts = {}

best_avg = float("-inf")
best_tile_seen = 0

NUM_GAMES = 550
for game in range(NUM_GAMES):

    score, tile, steps = play_one_game()

    scores.append(score)
    tiles.append(tile)
    steps_list.append(steps)

    tile_counts[tile] = (
        tile_counts.get(tile, 0) + 1
    )
    
    if (game + 1) % 25 == 0:
        current_avg = sum(scores) / (game + 1)

        if current_avg > best_avg:
            best_avg = current_avg
            torch.save(
                model.state_dict(),
                os.path.join(RESULT_DIR, "best_avg_model.pth")
            )

        if max(tiles) > best_tile_seen:
            best_tile_seen = max(tiles)
            torch.save(
                model.state_dict(),
                os.path.join(RESULT_DIR, "best_tile_model.pth")
            )
            
        target_model.load_state_dict(
            model.state_dict()
        )

        print(
            f"{game+1}/{NUM_GAMES}",
            "avg score =",
            sum(scores)/(game+1),
            "best tile =",
            max(tiles)
        )

torch.save(
    model.state_dict(),
    os.path.join(ROOT_DIR, "strong_2048_model.pth")
)

print("\nFINAL RESULTS")

print("Average Score:",
      sum(scores)/len(scores))

print("Best Score:",
      max(scores))

print("Average Tile:",
      sum(tiles)/len(tiles))

print("Best Tile:",
      max(tiles)
)

print("Average Steps:",
      sum(steps_list)/len(steps_list))

print(
    "Replay Buffer Size:",
    len(memory)
)

print("Tile Counts:")
print(tile_counts)

result_text = f"""
FINAL RESULTS

Average Score: {sum(scores) / len(scores)}
Best Score: {max(scores)}
Average Tile: {sum(tiles) / len(tiles)}
Best Tile: {max(tiles)}
Average Steps: {sum(steps_list) / len(steps_list)}
Replay Buffer Size: {len(memory)}

Tile Counts:
{tile_counts}
"""

result_txt_path = os.path.join(RESULT_DIR, "final_results.txt")

with open(result_txt_path, "w", encoding="utf-8") as f:
    f.write(result_text)

print("\nSaved final results to:")
print(result_txt_path)

torch.save(
    model.state_dict(),
    os.path.join(RESULT_DIR, "final_trained_model.pth")
)

print("Saved final trained model to:")
print(os.path.join(RESULT_DIR, "final_trained_model.pth"))

print("\nEVALUATING STRONG 2048 MODEL WITHOUT TRAINING...")

model.load_state_dict(
    torch.load(
        os.path.join(ROOT_DIR, "strong_2048_model.pth"),
        map_location="cpu"
    )
)

model.eval()

eval_scores = []
eval_tiles = []
eval_steps = []
eval_tile_counts = {}

EVAL_GAMES = 500

for game in range(EVAL_GAMES):
    env = Game2048Env()
    env.reset()
    done = False
    eval_steps_count = 0

    while not done:
        board_features = encode_board(env.board)

        x = torch.FloatTensor(
            board_features
        ).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)

        action = choose_action(
            env,
            logits,
            epsilon=0.0
        )

        _, _, done, _ = env.step(action)
        eval_steps_count += 1

    final_tile = max(
        max(row)
        for row in env.board
    )

    eval_scores.append(env.score)
    eval_tiles.append(final_tile)
    eval_steps.append(eval_steps_count)
    eval_tile_counts[final_tile] = (
        eval_tile_counts.get(final_tile, 0) + 1
    )

eval_2048_count = eval_tile_counts.get(2048, 0)

print("\nEVALUATION RESULTS")
print("Average Score:", sum(eval_scores) / len(eval_scores))
print("Best Score:", max(eval_scores))
print("Average Tile:", sum(eval_tiles) / len(eval_tiles))
print("Average Steps:", sum(eval_steps) / len(eval_steps))
print("Best Tile:", max(eval_tiles))
print("2048 Count:", eval_2048_count)
print("Tile Counts:")
print(eval_tile_counts)

evaluation_text = f"""
EVALUATION RESULTS

Average Score: {sum(eval_scores) / len(eval_scores)}
Best Score: {max(eval_scores)}
Average Tile: {sum(eval_tiles) / len(eval_tiles)}
Average Steps: {sum(eval_steps) / len(eval_steps)}
Best Tile: {max(eval_tiles)}
2048 Count: {eval_2048_count}

Tile Counts:
{eval_tile_counts}
"""

evaluation_txt_path = os.path.join(RESULT_DIR, "evaluation_results.txt")

with open(evaluation_txt_path, "w", encoding="utf-8") as f:
    f.write(evaluation_text)

print("\nSaved evaluation results to:")
print(evaluation_txt_path)