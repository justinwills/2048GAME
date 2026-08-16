"""
supervised_learning.py — Requirement 4
=======================================
Train a neural network (supervised learning) to play 2048.

Pipeline
--------
1. Data collection  : Run the evolved-weight expectimax AI to play N games,
                      recording (board_state → best_move) pairs.
2. Feature encoding : Each board cell value v is one-hot encoded over
                      16 possible tile classes (0, 2, 4, …, 32768) giving
                      a 4×4×16 = 256-dim input vector.
3. Model            : 3-layer MLP with BatchNorm + Dropout.
4. Training         : Cross-entropy loss, Adam optimiser, early stopping.
5. Evaluation       : Play games using the trained network; compare avg
                      score / max tile against heuristic baseline.

Usage
-----
    python supervised_learning.py           # full pipeline
    python supervised_learning.py --collect-only   # only generate dataset
    python supervised_learning.py --train-only     # train from saved dataset
    python supervised_learning.py --eval-only      # benchmark saved model
"""

import os, sys, json, argparse, random, math
import numpy as np

# ── ensure game_engine is importable ────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GAME_DIR   = os.path.join(SCRIPT_DIR, "2048 GAME NEW")
sys.path.insert(0, GAME_DIR)
sys.path.insert(0, SCRIPT_DIR)

from game_engine import (
    create_board, add_random_tile, can_move, moves,
    ai_move_with_weights, play_game_with_weights,
    count_empty, max_tile, copy_board, DEFAULT_WEIGHTS,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ── constants ────────────────────────────────────────────────────────────────
MOVE_KEYS   = ["w", "s", "a", "d"]          # up / down / left / right
MOVE_TO_IDX = {k: i for i, k in enumerate(MOVE_KEYS)}
TILE_CLASSES = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
TILE_TO_IDX  = {v: i for i, v in enumerate(TILE_CLASSES)}
INPUT_DIM    = 4 * 4 * len(TILE_CLASSES)    # 256
NUM_CLASSES  = 4

DATASET_PATH = os.path.join(SCRIPT_DIR, "sl_dataset.npz")
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "sl_model.pt")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "result", "sl_results.json")


# ════════════════════════════════════════════════════════════════════════════
# 1. Feature encoding
# ════════════════════════════════════════════════════════════════════════════

def encode_board(board: list) -> np.ndarray:
    """One-hot encode a 4×4 board → float32 array of shape (256,)."""
    vec = np.zeros((4, 4, len(TILE_CLASSES)), dtype=np.float32)
    for r in range(4):
        for c in range(4):
            v = board[r][c]
            idx = TILE_TO_IDX.get(v, TILE_TO_IDX[0])
            vec[r, c, idx] = 1.0
    return vec.flatten()


# ════════════════════════════════════════════════════════════════════════════
# 2. Data collection
# ════════════════════════════════════════════════════════════════════════════

def collect_dataset(n_games: int = 500, depth: int = 3,
                    weights: dict = None, seed: int = 42) -> tuple:
    """
    Play n_games with the evolved-weight expectimax and record every
    (board_state, chosen_move) pair.

    Returns
    -------
    X : np.ndarray  shape (N, 256) float32
    y : np.ndarray  shape (N,)     int64
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    random.seed(seed)
    X_list, y_list = [], []
    total_samples   = 0

    print(f"[Data Collection] Playing {n_games} games (depth={depth})…")

    for game_idx in range(n_games):
        board = create_board()
        add_random_tile(board)
        add_random_tile(board)
        score = 0

        while can_move(board):
            move_key = ai_move_with_weights(board, weights, depth=depth)
            if move_key is None:
                break

            # record this (state, label) pair
            X_list.append(encode_board(board))
            y_list.append(MOVE_TO_IDX[move_key])

            new_board, gained = moves[move_key](board)
            if new_board == board:
                break
            board  = new_board
            score += gained
            add_random_tile(board)

        total_samples += len(X_list) - total_samples
        if (game_idx + 1) % 50 == 0:
            print(f"  Game {game_idx + 1}/{n_games}  |  "
                  f"samples so far: {len(X_list):,}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.int64)
    print(f"[Data Collection] Done. Total samples: {len(X):,}")
    return X, y


def save_dataset(X, y, path=DATASET_PATH):
    np.savez_compressed(path, X=X, y=y)
    print(f"[Dataset] Saved → {path}  ({len(X):,} samples)")


def load_dataset(path=DATASET_PATH):
    data = np.load(path)
    print(f"[Dataset] Loaded ← {path}  ({len(data['X']):,} samples)")
    return data["X"], data["y"]


# ════════════════════════════════════════════════════════════════════════════
# 3. Neural network model
# ════════════════════════════════════════════════════════════════════════════

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
    def __init__(self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES):
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


# ════════════════════════════════════════════════════════════════════════════
# 4. Training
# ════════════════════════════════════════════════════════════════════════════

def train_model(X, y,
                epochs=40,
                batch_size=512,
                lr=1e-3,
                val_ratio=0.1,
                patience=8,
                seed=0) -> MoveNet:

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    # ── train / val split ───────────────────────────────────────────────────
    n          = len(X)
    idx        = rng.permutation(n)
    n_val      = int(n * val_ratio)
    val_idx    = idx[:n_val]
    train_idx  = idx[n_val:]

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx],   y[val_idx]

    print(f"[Training] train={len(X_tr):,}  val={len(X_va):,}")
    print(f"           epochs={epochs}  batch={batch_size}  lr={lr}")

    # ── class distribution ───────────────────────────────────────────────────
    counts = np.bincount(y_tr, minlength=4)
    print(f"[Training] Move distribution (train): "
          + "  ".join(f"{MOVE_KEYS[i]}={counts[i]:,}" for i in range(4)))

    # ── data loaders ─────────────────────────────────────────────────────────
    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr)
    X_va_t = torch.from_numpy(X_va)
    y_va_t = torch.from_numpy(y_va)

    tr_ds  = TensorDataset(X_tr_t, y_tr_t)
    va_ds  = TensorDataset(X_va_t, y_va_t)
    tr_dl  = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    va_dl  = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Device: {device}")

    model     = MoveNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_acc = 0.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        # ── train ────────────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for xb, yb in tr_dl:
            xb, yb   = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits   = model(xb)
            loss     = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            tr_loss    += loss.item() * len(xb)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += len(xb)

        tr_loss /= tr_total
        tr_acc   = tr_correct / tr_total

        # ── validate ─────────────────────────────────────────────────────────
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0

        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb   = xb.to(device), yb.to(device)
                logits   = model(xb)
                loss     = criterion(logits, yb)
                va_loss    += loss.item() * len(xb)
                va_correct += (logits.argmax(1) == yb).sum().item()
                va_total   += len(xb)

        va_loss /= va_total
        va_acc   = va_correct / va_total

        scheduler.step(va_acc)

        history.append({"epoch": epoch,
                         "tr_loss": round(tr_loss, 4),
                         "tr_acc":  round(tr_acc,  4),
                         "va_loss": round(va_loss, 4),
                         "va_acc":  round(va_acc,  4)})

        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
              f"va_loss={va_loss:.4f}  va_acc={va_acc:.3f}"
              + ("  ← best" if va_acc > best_val_acc else ""))

        if va_acc > best_val_acc:
            best_val_acc       = va_acc
            epochs_no_improve  = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Training] Early stopping at epoch {epoch}.")
                break

    print(f"[Training] Best val accuracy: {best_val_acc:.3f}")
    # restore best checkpoint
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    return model, history, best_val_acc


# ════════════════════════════════════════════════════════════════════════════
# 5. Evaluation — play games with the trained network
# ════════════════════════════════════════════════════════════════════════════

def nn_move(model, board, device):
    """Select a move using the neural network. Falls back to first valid move."""
    model.eval()
    with torch.no_grad():
        x      = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
        logits = model(x).squeeze(0)
        # sort moves by predicted probability; pick first valid one
        order  = torch.argsort(logits, descending=True).tolist()

    for idx in order:
        key = MOVE_KEYS[idx]
        new_board, _ = moves[key](board)
        if new_board != board:
            return key
    return None


def play_game_nn(model, device, seed=None) -> tuple:
    """Play one full game with the neural network. Returns (score, max_tile, steps)."""
    if seed is not None:
        random.seed(seed)

    board = create_board()
    add_random_tile(board)
    add_random_tile(board)
    score, steps = 0, 0

    while can_move(board):
        move_key = nn_move(model, board, device)
        if move_key is None:
            break
        new_board, gained = moves[move_key](board)
        if new_board == board:
            break
        board  = new_board
        score += gained
        steps += 1
        add_random_tile(board)

    return score, max_tile(board), steps


def evaluate_nn(model, device, n_games=100, seed=42) -> dict:
    """Benchmark the neural network over n_games."""
    random.seed(seed)
    scores, tiles, steps_list = [], [], []
    tile_counts = {}

    print(f"\n[NN Eval] Playing {n_games} games…")
    for i in range(n_games):
        s, t, st = play_game_nn(model, device)
        scores.append(s);  tiles.append(t);  steps_list.append(st)
        tile_counts[t] = tile_counts.get(t, 0) + 1
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_games}  running avg score = {sum(scores)/(i+1):.0f}")

    result = {
        "avg_score":  round(sum(scores)  / n_games, 1),
        "best_score": max(scores),
        "avg_tile":   round(sum(tiles)   / n_games, 1),
        "best_tile":  max(tiles),
        "avg_steps":  round(sum(steps_list) / n_games, 1),
        "tile_distribution": {str(k): v for k, v in sorted(tile_counts.items())},
    }
    return result


def evaluate_heuristic(n_games=100, seed=42) -> dict:
    """Baseline: evolved-weight expectimax (depth 3)."""
    random.seed(seed)
    scores, tiles, steps_list = [], [], []
    tile_counts = {}

    print(f"\n[Heuristic Eval] Playing {n_games} games (expectimax depth=3)…")
    for i in range(n_games):
        s, t, st = play_game_with_weights(DEFAULT_WEIGHTS, depth=3, show=False)
        scores.append(s);  tiles.append(t);  steps_list.append(st)
        tile_counts[t] = tile_counts.get(t, 0) + 1
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_games}  running avg score = {sum(scores)/(i+1):.0f}")

    result = {
        "avg_score":  round(sum(scores)  / n_games, 1),
        "best_score": max(scores),
        "avg_tile":   round(sum(tiles)   / n_games, 1),
        "best_tile":  max(tiles),
        "avg_steps":  round(sum(steps_list) / n_games, 1),
        "tile_distribution": {str(k): v for k, v in sorted(tile_counts.items())},
    }
    return result


# ════════════════════════════════════════════════════════════════════════════
# 6. Pretty-print helpers
# ════════════════════════════════════════════════════════════════════════════

def print_results(nn_res, heuristic_res, history, best_val_acc):
    sep = "=" * 60
    print(f"\n{sep}")
    print("REQUIREMENT 4 — SUPERVISED LEARNING RESULTS")
    print(sep)

    print("\n── Training ─────────────────────────────────────────────")
    print(f"  Training samples : {sum(1 for _ in history):,} epochs run")
    print(f"  Best val accuracy: {best_val_acc:.3f} "
          f"({best_val_acc*100:.1f}%)")

    print("\n── Game Performance (100 games each) ────────────────────")
    header = f"  {'Metric':<22} {'NN':>10} {'Heuristic':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    metrics = [
        ("Avg Score",      "avg_score"),
        ("Best Score",     "best_score"),
        ("Avg Max Tile",   "avg_tile"),
        ("Best Max Tile",  "best_tile"),
        ("Avg Steps",      "avg_steps"),
    ]
    for label, key in metrics:
        nn_v = nn_res[key]
        he_v = heuristic_res[key]
        print(f"  {label:<22} {nn_v:>10}  {he_v:>10}")

    print("\n── NN Tile Distribution ─────────────────────────────────")
    for tile, cnt in sorted(nn_res["tile_distribution"].items(), key=lambda x: int(x[0])):
        bar = "█" * int(cnt)
        print(f"  {int(tile):>6} : {cnt:>3}x  {bar}")

    print("\n── Heuristic Tile Distribution ──────────────────────────")
    for tile, cnt in sorted(heuristic_res["tile_distribution"].items(), key=lambda x: int(x[0])):
        bar = "█" * int(cnt)
        print(f"  {int(tile):>6} : {cnt:>3}x  {bar}")

    if heuristic_res["avg_score"] > 0:
        pct = (nn_res["avg_score"] - heuristic_res["avg_score"]) / heuristic_res["avg_score"] * 100
        print(f"\n  Score vs heuristic: {pct:+.1f}%")
    print(sep)


# ════════════════════════════════════════════════════════════════════════════
# 7. Main entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Requirement 4 – Supervised Learning")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--train-only",   action="store_true")
    parser.add_argument("--eval-only",    action="store_true")
    parser.add_argument("--games",        type=int, default=500,
                        help="Games for data collection (default 500)")
    parser.add_argument("--eval-games",   type=int, default=100,
                        help="Games for final evaluation (default 100)")
    parser.add_argument("--epochs",       type=int, default=40)
    parser.add_argument("--batch-size",   type=int, default=512)
    parser.add_argument("--depth",        type=int, default=3,
                        help="Expectimax depth for data collection")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── step 1: collect ──────────────────────────────────────────────────────
    if not args.train_only and not args.eval_only:
        X, y = collect_dataset(n_games=args.games, depth=args.depth)
        save_dataset(X, y)
    if args.collect_only:
        return

    # ── step 2: train ────────────────────────────────────────────────────────
    if not args.eval_only:
        X, y = load_dataset()
        model, history, best_val_acc = train_model(
            X, y, epochs=args.epochs, batch_size=args.batch_size
        )
    else:
        model = MoveNet()
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model = model.to(device)
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH) as _f:
                _prev = json.load(_f)
            history      = _prev.get("training", {}).get("history", [])
            best_val_acc = _prev.get("training", {}).get("best_val_acc", 0.0)
        else:
            history, best_val_acc = [], 0.0
        print(f"[Eval] Loaded model from {MODEL_PATH}")
    if args.train_only:
        return

    # ── step 3: evaluate ─────────────────────────────────────────────────────
    model = model.to(device)
    nn_res         = evaluate_nn(model, device, n_games=args.eval_games)
    heuristic_res  = evaluate_heuristic(n_games=args.eval_games)

    print_results(nn_res, heuristic_res, history, best_val_acc)

    results = {
        "nn":            nn_res,
        "heuristic":     heuristic_res,
        "training": {
            "best_val_acc": round(best_val_acc, 4),
            "epochs_run":   len(history),
            "history":      history,
        }
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results] Saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
