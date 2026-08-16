"""
Requirement 3: Evolutionary Computation Optimization for 2048 Heuristic Weights
================================================================================
Uses a (μ + λ) Evolution Strategy with self-adaptive sigma (CMA-ES-lite style)
to optimize the 5 evaluation weights from 2048.py:
  - empty   : reward for free cells
  - biggest : reward for the max tile value
  - smooth  : penalty weight for tile value differences between neighbors
  - corner  : bonus multiplier for keeping the max tile in a corner
  - mono    : reward for monotone rows/columns

Strategy:
  - Population size μ = 12  (parents)
  - Offspring per gen λ = 36 (offspring = 3×μ, standard ES ratio)
  - Each individual = [empty, biggest, smooth, corner, mono] + 5 sigmas
  - Fitness = average score over EVAL_GAMES games (with fixed depth)
  - Selection: (μ + λ) — best μ from parents ∪ offspring survive
  - Termination: MAX_GENERATIONS or patience (no improvement for PATIENCE gens)

Run:
    python evolve.py

Outputs:
    - evolution_log.csv   : per-generation stats
    - best_weights.json   : best weights found
    - evolution_plot.png  : fitness curve (if matplotlib available)
"""

import random
import math
import copy
import json
import csv
import time
import sys
import os
from pathlib import Path

# ── import game engine from 2048.py ─────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from game_engine import (
    BASELINE_WEIGHTS,
    create_board, add_random_tile, can_move,
    moves, evaluate, ai_move_with_weights, play_game_with_weights
)

# ── ES hyper-parameters ──────────────────────────────────────────────────────
MU          = 12       # parent pool size
LAMBDA      = 36       # offspring per generation
MAX_GEN     = 60       # max generations
EVAL_GAMES  = 8        # games per fitness evaluation (speed vs accuracy trade-off)
EVAL_DEPTH  = 3        # expectimax search depth during evolution (keep low for speed)
PATIENCE    = 15       # stop if no improvement for this many generations
SEED        = 42

random.seed(SEED)

# ── weight search space ──────────────────────────────────────────────────────
#   (lower_bound, upper_bound, initial_sigma)
PARAM_SPACE = {
    "empty":   (100,   5000,  300.0),
    "biggest": (0.5,   20.0,    2.0),
    "smooth":  (0.05,   5.0,    0.5),
    "corner":  (1.0,   30.0,    3.0),
    "mono":    (0.5,   20.0,    2.0),
}
PARAM_KEYS = list(PARAM_SPACE.keys())

# ── individual representation ────────────────────────────────────────────────

def random_individual():
    """Create a random individual with random weights and initial sigmas."""
    weights = {}
    sigmas  = {}
    for key, (lo, hi, sig0) in PARAM_SPACE.items():
        weights[key] = random.uniform(lo, hi)
        sigmas[key]  = sig0
    return {"weights": weights, "sigmas": sigmas, "fitness": None}


def individual_from_weights(weights):
    """Create an individual from a known weight set."""
    sigmas = {key: PARAM_SPACE[key][2] for key in PARAM_KEYS}
    return {"weights": clip_weights(weights), "sigmas": sigmas, "fitness": None}


def clip_weights(weights):
    """Keep weights inside the search bounds."""
    clipped = {}
    for key, (lo, hi, _) in PARAM_SPACE.items():
        clipped[key] = max(lo, min(hi, weights[key]))
    return clipped


# ── self-adaptive mutation (uncorrelated, one-sigma-per-parameter) ───────────

TAU  = 1.0 / math.sqrt(2 * len(PARAM_KEYS))        # global learning rate
TAU_ = 1.0 / math.sqrt(2 * math.sqrt(len(PARAM_KEYS)))  # local learning rate
EPS  = 1e-6  # minimum sigma

def mutate(ind):
    """
    Self-adaptive ES mutation:
      σ'_i = σ_i · exp(τ' · N(0,1) + τ · N_i(0,1))
      x'_i = x_i + σ'_i · N_i(0,1)
    """
    global_factor = math.exp(TAU_ * random.gauss(0, 1))
    new_weights   = {}
    new_sigmas    = {}

    for key in PARAM_KEYS:
        local_factor   = math.exp(TAU * random.gauss(0, 1))
        new_sig        = max(EPS, ind["sigmas"][key] * global_factor * local_factor)
        new_val        = ind["weights"][key] + new_sig * random.gauss(0, 1)
        new_sigmas[key] = new_sig
        new_weights[key] = new_val

    child = {
        "weights": clip_weights(new_weights),
        "sigmas":  new_sigmas,
        "fitness": None,
    }
    return child


def recombine(p1, p2):
    """
    Intermediate (arithmetic) recombination for weights;
    discrete recombination for sigmas — standard ES practice.
    """
    child_w = {}
    child_s = {}
    for key in PARAM_KEYS:
        child_w[key] = 0.5 * (p1["weights"][key] + p2["weights"][key])
        child_s[key] = random.choice([p1["sigmas"][key], p2["sigmas"][key]])
    return {
        "weights": clip_weights(child_w),
        "sigmas":  child_s,
        "fitness": None,
    }


# ── fitness evaluation ────────────────────────────────────────────────────────

def evaluate_individual(ind, n_games=EVAL_GAMES, depth=EVAL_DEPTH):
    """Average score over n_games as fitness."""
    total = 0
    for _ in range(n_games):
        score, _, _ = play_game_with_weights(ind["weights"], depth=depth)
        total += score
    ind["fitness"] = total / n_games
    return ind["fitness"]


# ── (μ + λ) Evolution Strategy ───────────────────────────────────────────────

def run_evolution():
    print("=" * 60)
    print("  2048 Weight Optimization — (μ+λ) Evolution Strategy")
    print(f"  μ={MU}  λ={LAMBDA}  max_gen={MAX_GEN}  eval_games={EVAL_GAMES}  depth={EVAL_DEPTH}")
    print("=" * 60)

    # ── initialise population ────────────────────────────────────────────────
    population = [individual_from_weights(BASELINE_WEIGHTS)]
    population.extend(random_individual() for _ in range(MU - 1))

    print("\n[Init] Evaluating initial population …")
    for idx, ind in enumerate(population):
        evaluate_individual(ind)
        print(f"  Individual {idx+1:2d}: fitness={ind['fitness']:.1f}  "
              f"weights={fmt_weights(ind['weights'])}")

    best_ever    = max(population, key=lambda x: x["fitness"])
    best_ever    = copy.deepcopy(best_ever)
    no_improve   = 0

    log_rows = []

    for gen in range(1, MAX_GEN + 1):
        t0 = time.time()

        # ── generate λ offspring ─────────────────────────────────────────────
        offspring = []
        for _ in range(LAMBDA):
            if random.random() < 0.5 and MU >= 2:
                # recombination + mutation
                p1, p2 = random.sample(population, 2)
                child  = recombine(p1, p2)
                child  = mutate(child)
            else:
                # mutation only
                parent = random.choice(population)
                child  = mutate(parent)
            offspring.append(child)

        # ── evaluate offspring ───────────────────────────────────────────────
        for child in offspring:
            evaluate_individual(child)

        # ── (μ + λ) selection ────────────────────────────────────────────────
        combined   = population + offspring
        combined.sort(key=lambda x: x["fitness"], reverse=True)
        population = combined[:MU]

        # ── stats ─────────────────────────────────────────────────────────────
        gen_best   = population[0]
        gen_avg    = sum(x["fitness"] for x in population) / MU
        gen_worst  = population[-1]["fitness"]
        elapsed    = time.time() - t0

        improved = gen_best["fitness"] > best_ever["fitness"]
        if improved:
            best_ever  = copy.deepcopy(gen_best)
            no_improve = 0
            flag = " ★ NEW BEST"
        else:
            no_improve += 1
            flag = ""

        print(f"\n[Gen {gen:3d}] best={gen_best['fitness']:.1f}  "
              f"avg={gen_avg:.1f}  worst={gen_worst:.1f}  "
              f"time={elapsed:.1f}s{flag}")
        print(f"         weights={fmt_weights(gen_best['weights'])}")

        log_rows.append({
            "generation":  gen,
            "best":        gen_best["fitness"],
            "avg":         gen_avg,
            "worst":       gen_worst,
            **{f"w_{k}": gen_best["weights"][k] for k in PARAM_KEYS},
        })

        if no_improve >= PATIENCE:
            print(f"\n[Early stop] No improvement for {PATIENCE} generations.")
            break

    return best_ever, log_rows


# ── helpers ───────────────────────────────────────────────────────────────────

def fmt_weights(w):
    return "  ".join(f"{k}={w[k]:.2f}" for k in PARAM_KEYS)


def save_results(best, log_rows, benchmark_summary=None):
    # best weights JSON
    out = {"weights": best["weights"], "fitness": best["fitness"]}
    if benchmark_summary is not None:
        out["benchmark"] = benchmark_summary
    with open(SCRIPT_DIR / "best_weights.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n[Saved] best_weights.json")

    # evolution log CSV
    if log_rows:
        with open(SCRIPT_DIR / "evolution_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        print("[Saved] evolution_log.csv")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gens  = [r["generation"] for r in log_rows]
        bests = [r["best"]       for r in log_rows]
        avgs  = [r["avg"]        for r in log_rows]

        plt.figure(figsize=(10, 5))
        plt.plot(gens, bests, label="Best fitness",    linewidth=2)
        plt.plot(gens, avgs,  label="Avg fitness",     linewidth=2, linestyle="--")
        plt.xlabel("Generation")
        plt.ylabel("Avg Score (fitness)")
        plt.title("Evolution Strategy — 2048 Weight Optimization")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(SCRIPT_DIR / "evolution_plot.png", dpi=150)
        print("[Saved] evolution_plot.png")
    except ImportError:
        print("[Skip]  matplotlib not available — no plot saved.")


def benchmark(weights, label, n=30, depth=4):
    """Run n full games with given weights and print statistics."""
    scores, tiles, steps = [], [], []
    tile_counts = {}
    print(f"\n{'='*60}")
    print(f"  Benchmark: {label}  (n={n}, depth={depth})")
    print(f"  Weights: {fmt_weights(weights)}")
    print(f"{'='*60}")
    for i in range(n):
        s, t, st = play_game_with_weights(weights, depth=depth)
        scores.append(s); tiles.append(t); steps.append(st)
        tile_counts[t] = tile_counts.get(t, 0) + 1
        print(f"  Game {i+1:3d}: score={s:6d}  max_tile={t:5d}  steps={st}")
    print(f"\n  Avg Score : {sum(scores)/n:.1f}")
    print(f"  Best Score: {max(scores)}")
    print(f"  Avg Tile  : {sum(tiles)/n:.1f}")
    print(f"  Best Tile : {max(tiles)}")
    print(f"  Avg Steps : {sum(steps)/n:.1f}")
    print("\n  Tile distribution:")
    for t in sorted(tile_counts):
        print(f"    {t:5d}: {tile_counts[t]:3d} games ({tile_counts[t]/n*100:.1f}%)")
    return sum(scores)/n


def benchmark_summary(original_avg, evolved_avg, n=30, depth=4):
    return {
        "baseline_avg_score": original_avg,
        "evolved_avg_score": evolved_avg,
        "improvement_percent": (evolved_avg - original_avg) / original_avg * 100,
        "games": n,
        "depth": depth,
    }


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    best, log_rows = run_evolution()

    print("\n" + "=" * 60)
    print("  EVOLUTION COMPLETE")
    print(f"  Best fitness (avg score over {EVAL_GAMES} games): {best['fitness']:.1f}")
    print(f"  Best weights: {fmt_weights(best['weights'])}")
    print("=" * 60)

    # ── final benchmark: original weights vs evolved weights ─────────────────
    original_weights = BASELINE_WEIGHTS

    print("\n\n[Benchmark] Comparing original vs evolved weights over 30 games …")
    avg_orig  = benchmark(original_weights,   "Original weights",  n=30, depth=4)
    avg_evol  = benchmark(best["weights"],    "Evolved weights",   n=30, depth=4)

    save_results(best, log_rows, benchmark_summary(avg_orig, avg_evol))

    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"  Original avg score : {avg_orig:.1f}")
    print(f"  Evolved  avg score : {avg_evol:.1f}")
    improvement = (avg_evol - avg_orig) / avg_orig * 100
    print(f"  Improvement        : {improvement:+.1f}%")
    print(f"{'='*60}")
