# 2048 AI Project

## Overview

This project implements several AI methods for playing 2048:

1. Heuristic Search (Expectimax)
2. Evolutionary Computation for heuristic optimization
3. Supervised Learning using Neural Networks
4. Reinforcement Learning (DQN-based training)

---

## Project Structure

```text
2048GAME/
├── 2048.py
├── env_2048.py
├── evolutionary_computation.py
├── evolve.py
├── game2048.py
├── game_engine.py
├── generate_dataset.py
├── supervised_learning.py
├── training_2048.py
├── dataset.csv
├── sl_dataset.npz
├── training_log.csv
├── requirements.txt
├── README.md
│
│   
├── models/
│   ├── best_rl_model.pth
│   ├── dqn_2048.pth
│   ├── model_1024.pth
│   ├── model_2048.pth
│   ├── sl_model.pt
│   └── teacher_model.pth
│
│
├── result/
│   ├── best_avg_model.pth
│   ├── best_tile_model.pth
│   ├── best_weights.json
│   ├── evaluation_results.txt
│   ├── final_results.txt
│   └── sl_results.json
│
├── web_demo/
│   ├── index.html
│   ├── ai.js
│   ├── game.js
│   └── style.css
│
└── 报告/
    ├── 大作业报告.docx
    ├── 大作业报告.pdf
```
---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ is recommended.

---

## 1. Heuristic Search and Evolutionary Computation

Run heuristic evaluation:

```bash
python evolutionary_computation.py
```

Run evolutionary optimization:

```bash
python evolve.py
```

Output is saved in result/best_weights.json and result/
Results are saved in result/sl_results.json.

---

## 2. Dataset Generation

```bash
python generate_dataset.py
```

Outputs:

- `dataset.csv`

---

## 3. Supervised Learning

Train the neural network:

```bash
python supervised_learning.py
```

Evaluate:

```bash
python supervised_learning.py --eval-only
```
If evaluation is slow on CPU, run:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python supervised_learning.py --eval-only

Model file: `models/sl_model.pt`

Results are saved in `sl_results.json`.

---

## 4. Reinforcement Learning

Train RL agent:

```bash
python training_2048.py
```

Requires `env_2048.py`.  

Trained models:

- `models/best_rl_model.pth`
- `models/model_2048.pth`

Evaluation results in `result/`.

---

## 5. Web Demo

You can run a web version of 2048 with AI:

1. Open the `web_demo/index.html` file in a browser.
2. The AI logic is in `web_demo/ai.js`.
3. The game logic is in `web_demo/game.js`.
4. CSS styling is in `web_demo/style.css`.

No server setup is needed; it runs locally in your browser.

---

## Results

Example result files:

- `result/evaluation_results.txt`
- `result/final_results.txt`
- `sl_results.json`
- `best_weights.json`

---

## Notes

Ensure model files are in `models/`. If you move them, update the path in:

- `supervised_learning.py`
- `training_2048.py`

---