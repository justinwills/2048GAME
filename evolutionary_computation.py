"""
Requirement 3: benchmark evolved heuristic weights for 2048.

This file is intentionally separate from 2048.py so Requirement 3 can be
tested directly:

    python evolutionary_computation.py
    python evolutionary_computation.py --games 20 --depth 3 --seed 123
"""

import argparse
import json
import random
from pathlib import Path

from game_engine import BASELINE_WEIGHTS, DEFAULT_WEIGHTS, play_game_with_weights


SCRIPT_DIR = Path(__file__).resolve().parent
BEST_WEIGHTS_PATH = SCRIPT_DIR / "result" / "best_weights.json"


def load_evolved_weights(path=BEST_WEIGHTS_PATH):
    """Load evolved weights produced by evolve.py, or use the checked-in default."""
    if not Path(path).exists():
        return DEFAULT_WEIGHTS.copy()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    weights = data.get("weights")
    if not isinstance(weights, dict):
        raise ValueError(f"{path} does not contain a valid 'weights' object")

    required = set(BASELINE_WEIGHTS)
    missing = required - set(weights)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing weight(s): {missing_list}")

    return {key: float(weights[key]) for key in BASELINE_WEIGHTS}


def benchmark_weights(label, weights, games=10, depth=3, seed=123):
    random.seed(seed)
    scores = []
    max_tiles = []
    steps_list = []
    tile_counts = {}

    for _ in range(games):
        score, tile, steps = play_game_with_weights(weights, depth=depth, show=False)
        scores.append(score)
        max_tiles.append(tile)
        steps_list.append(steps)
        tile_counts[tile] = tile_counts.get(tile, 0) + 1

    result = {
        "avg_score": sum(scores) / games,
        "best_score": max(scores),
        "avg_tile": sum(max_tiles) / games,
        "best_tile": max(max_tiles),
        "avg_steps": sum(steps_list) / games,
        "tile_counts": dict(sorted(tile_counts.items())),
    }

    print(f"\n{label}")
    print(f"Average Score: {result['avg_score']:.1f}")
    print(f"Best Score: {result['best_score']}")
    print(f"Average Max Tile: {result['avg_tile']:.1f}")
    print(f"Best Max Tile: {result['best_tile']}")
    print(f"Average Steps: {result['avg_steps']:.1f}")
    print("Tile Distribution:", result["tile_counts"])

    return result


def benchmark_evolutionary_computation(games=10, depth=3, seed=123, evolved_weights=None):
    if games <= 0:
        raise ValueError("games must be greater than 0")
    if depth <= 0:
        raise ValueError("depth must be greater than 0")

    evolved_weights = evolved_weights or load_evolved_weights()

    print("===== REQUIREMENT 3 BENCHMARK =====")
    print("Evolutionary computation optimized the heuristic evaluation weights.")
    print(f"Games: {games}, Depth: {depth}, Seed: {seed}")

    baseline = benchmark_weights(
        "Baseline weights",
        BASELINE_WEIGHTS,
        games=games,
        depth=depth,
        seed=seed,
    )
    evolved = benchmark_weights(
        "Evolved weights",
        evolved_weights,
        games=games,
        depth=depth,
        seed=seed,
    )

    improvement = (
        (evolved["avg_score"] - baseline["avg_score"]) / baseline["avg_score"] * 100
    )

    print("\n===== REQUIREMENT 3 RESULT =====")
    print(f"Baseline Average Score: {baseline['avg_score']:.1f}")
    print(f"Evolved Average Score: {evolved['avg_score']:.1f}")
    print(f"Improvement: {improvement:+.1f}%")
    print(f"Baseline Best Tile: {baseline['best_tile']}")
    print(f"Evolved Best Tile: {evolved['best_tile']}")

    return {
        "baseline": baseline,
        "evolved": evolved,
        "improvement_percent": improvement,
        "games": games,
        "depth": depth,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Evolutionary computation benchmark")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    benchmark_evolutionary_computation(games=args.games, depth=args.depth, seed=args.seed)


if __name__ == "__main__":
    main()
