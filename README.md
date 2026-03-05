# Gomoku 9x9 Policy Learning System (Supervised Baseline)

## 0) Repository purpose

This repository implements a complete Python pipeline for **9x9 Gomoku** with end-to-end
supervised policy learning:

1. deterministic game environment and rule engine,
2. supervised data generation from heuristic and model teachers,
3. checkpointed CNN policy training with hard or soft labels,
4. tactical and outcome aware sample reweighting,
5. symmetry augmentation and robust reproducibility,
6. terminal and GUI play interfaces.

The objective is a practical, low-dependency baseline in which model behavior evolves
from weak policy imitation to a competitive heuristic-competitive opponent.

**See also:** [`CHANGELOG.md`](CHANGELOG.md) for a chronological experiment ledger.

---

## 1) Technical architecture

```text
gomoku/
├── README.md
├── docs/
│   ├── training_tips.md
│   └── reproducibility.md
├── CHANGELOG.md
├── requirements.txt
├── main.py
├── src/
│   ├── __init__.py
│   ├── env/
│   │   ├── __init__.py
│   │   ├── gomoku_env.py
│   │   └── rules.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── random_agent.py
│   │   ├── heuristic_agent.py
│   │   └── model_agent.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── game_logging.py
│   │   └── selfplay_generate.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── policy_net.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_supervised.py
│   └── utils/
│       ├── __init__.py
│       └── encoding.py
├── tests/
│   ├── test_rules.py
│   ├── test_game_logging.py
│   └── test_outcome_weights.py
```

## 1.1) Additional documentation

- [`docs/training_tips.md`](docs/training_tips.md): practical training and iteration command cookbook.
- [`docs/reproducibility.md`](docs/reproducibility.md): experiment protocol, split governance, seed strategy, and reproducibility checks.

---

## 2) State, action, and reward formalism

- Board: fixed 9x9 integer array `board[r, c]`
  - `1` = black
  - `-1` = white
  - `0` = empty
- Action: `a = r * 9 + c`, where `a in [0, 80]`
- Current player sign `p_t in {-1, 1}`
- Reward semantics in `step(action)`:
  - terminal win for mover at turn `t`: `+1.0` (from mover perspective)
  - draw: `0.0`
  - non-terminal: `0.0`

`src/env/gomoku_env.py` enforces legality, win checks, side switching, and terminal-state guards.

---

## 3) Policy encoding from current-player perspective

`src/utils/encoding.py` returns `np.float32` tensor shape `(3, 9, 9)`:

1. channel-0: current-player stones
2. channel-1: opponent stones
3. channel-2: turn plane (`1.0` if current player is black, else `0.0`)

This encoding is **always normalized to current player perspective** so that
`+1` and `-1` semantics remain invariant to absolute color and simplify network learning.

---

## 4) Rule module and threat heuristics

`src/env/rules.py` holds all game logic:

- `check_five_in_a_row(board, row, col, player, win_length=5)`
- `count_in_direction(board, row, col, player, dr, dc)`
- `board_full(board)`
- tactical helpers:
  - `immediate_winning_actions`
  - `max_player_threat_score`
  - `action_threat_score`
  - `local_candidate_actions`

Threat scoring is directional and coarse-quantized with
`pattern_score(length, open_ends)` where:

- `length` = number of contiguous stones that can appear in a line fragment for the player,
- `open_ends` = available extension cells on both sides,
- higher `length` and more `open_ends` produce larger threat scores.

This is used as a surrogate proxy for tactical urgency (e.g., immediate win, forced defense,
high-value build threat, etc.) in both generator and training weighting.

---

## 5) Model

`src/models/policy_net.py` defines `PolicyNet`.

- input: `(B, 3, 9, 9)`
- conv stack:
  - `Conv2d(3, 32, k=3, p=1) -> ReLU`
  - `Conv2d(32, 64, k=3, p=1) -> ReLU`
  - `Conv2d(64, 64, k=3, p=1) -> ReLU`
- flatten
- `Linear(64*9*9, 256) -> ReLU`
- `Linear(256, 81)` logits

Output are unnormalized logits `z ∈ R^81` interpreted as action scores.
Invalid moves are masked to very negative values before softmax/argmax.

---

## 6) Data generation and logging

### 6.1 Self-play generation (`src/data/selfplay_generate.py`)

Each move stores:

- `state`: encoded `(3, 9, 9)` tensor,
- `action`: hard label in `[0, 80]`,
- optional `policy_targets`: soft labels `(81,)` from teacher policy,
- optional `outcomes`: `1/-1/0` per move (winner/loser/draw from game result).

Optional teacher mode flags:
- `--teacher-checkpoint`, `--teacher-device`
- `--teacher-noise`
- `--teacher-temperature`
- `--teacher-hard-mix`
- `--save-soft-targets`

### 6.2 Human-game logging (`src/data/game_logging.py`)

`main.py` can append human sessions and model/heuristic games into `.npz` dataset format.
Logged artifacts are immediately usable by training and preserve outcome metadata so winning
and losing trajectories can be reweighted differently.

---

## 7) Agents

### 7.1 Heuristic baseline
`HeuristicAgent` applies a deterministic ladder by default:

1. take immediate winning move,
2. block opponent immediate win,
3. create strongest own threat,
4. suppress strongest opponent threat,
5. center prior,
6. near-stone prior (within Chebyshev radius),
7. fallback policy.

Small optional action-noise exists for data diversification.

### 7.2 Learned policy agent
`ModelAgent`:

- loads checkpoint formats (`raw_state_dict`, `state_dict`, `model_state_dict`),
- encodes state from side-to-move,
- masks illegal actions,
- applies tactical guard logic if no valid high-confidence soft policy exists,
- chooses `argmax` policy index.

### 7.3 Random baseline
Simple legal uniform random policy for sanity checks and baseline comparisons.

---

## 8) Training mathematics and objective

### 8.1 Hard target cross-entropy
For sample `i` with action label `y_i`, logits `z_i`:

`L_hard(i) = - log( exp(z_i[y_i]) / sum_j exp(z_i[j]) )`

### 8.2 Soft target KL/CE form
For target distribution `p_i` (row-stochastic):

`L_soft(i) = - sum_a p_i[a] * log( exp(z_i[a]) / sum_j exp(z_i[j]) )`

In practice this is implemented by cross-entropy with per-sample soft probabilities.

### 8.3 Sample weighting
Each sample receives weight `w_i`:

`w_i = w_tactical_i * w_outcome_i`

Weighted batch loss (both train and validation):

`L_batch = ( Σ_i w_i * L_i ) / ( Σ_i w_i )`

Tactical weights (`w_tactical`) include user-configured multipliers for:

- immediate own win (`attack-weight`),
- must-block (`defense-weight`),
- threat suppression (`threat-weight`),
- build-up threat (`build-threat-weight`).

Outcome weights (`w_outcome`) depend on move-level game result:

- winner-side moves -> `outcome-win-weight`,
- loser-side moves -> `outcome-loss-weight`,
- draw moves -> `outcome-draw-weight`.

### 8.4 Optimization and regularization

- optimizer: Adam
- scheduler: `ReduceLROnPlateau`
- early stop on `val_loss` plateau
- consistent metrics: top-1 and top-5 accuracy on legal labels over full action space
- optional fixed split persistence via `--split-path`

### 8.5 Symmetry augmentation
If enabled, each sampled state is transformed using the 8-fold dihedral group `D4`
(rotations and reflections). Action indices and optional soft targets are remapped using
precomputed transform tables per sample.

---

## 9) Metric interpretation

For 81-way uniform random policy, expected CE baseline is ~`ln(81) = 4.3944`.
So values in early training around `3+` indicate weak signal; sub-1.x is strong for this baseline task,
while values near `0.7` are usually associated with much larger datasets, strong augmentation,
and low-label-noise teachers.

---

## 10) Important CLI commands

### 10.1 Generate soft-label self-play data from latest policy

```bash
python -m src.data.selfplay_generate --num-games 1000 --output data/selfplay_teacher_1k_soft.npz --seed 42 --noise 0.05 --teacher-checkpoint checkpoints/policy_net.pt --teacher-device cuda --teacher-temperature 1.0 --teacher-hard-mix 0.15 --save-soft-targets
```

### 10.2 Mix human and self-play data and train with outcome+threat-aware supervision

```bash
python -m src.training.train_supervised --data data/selfplay_teacher_1k_soft.npz --extra-data data/human_games.npz --target-mode auto --epochs 120 --batch-size 128 --lr 0.0003 --device cuda --save-path checkpoints/policy_net.pt --symmetry-augment --split-path checkpoints/train_val_split_outcome.npz --outcome-win-weight 1.5 --outcome-loss-weight 1.2 --outcome-draw-weight 1.0 --defense-weight 3.0 --attack-weight 3.0 --threat-weight 1.8 --build-threat-weight 2.0 --threat-threshold 2000 --weight-scan-radius 2 --weight-progress-every 1000 --resume-from checkpoints/policy_net.pt
```

### 10.3 Resume training with improved split and same seed

```bash
python -m src.training.train_supervised --data data/selfplay_teacher_1k_soft.npz --extra-data data/human_games.npz --target-mode auto --epochs 60 --batch-size 128 --lr 0.0003 --device cuda --save-path checkpoints/policy_net.pt --split-path checkpoints/train_val_split_outcome.npz --resume-from checkpoints/policy_net.pt --seed 42
```

### 10.4 Play in terminal

```bash
python main.py --mode human-vs-model --model-path checkpoints/policy_net.pt --human-player black --device cuda
```

### 10.5 Play in GUI

```bash
python main.py --ui gui --mode human-vs-model --model-path checkpoints/policy_net.pt --human-player black --device cuda
```

---

## 11) Experiment ledger (transition to competitive behavior)

| Date (UTC+?) | Phase | Core change | Evidence |
|---|---|---|---|
| 2026-03-03 | Baseline | Environment + hard-label heuristic self-play only | Very high val loss (~4.x), weak tactical consistency |
| 2026-03-03 | Teacher distillation | Added `--teacher-checkpoint`, soft target export | Smoother optimization dynamics, better generalization under sparse labels |
| 2026-03-04 | Tactical weighting | Added threat-scored sample amplification and per-move tactical flags | Improved defensive quality (fewer immediate tactical misses) |
| 2026-03-04 | Reproducible split + resume + scheduler | Added `--split-path`, `--resume-from`, LR scheduling | Stable later-phase convergence, reduced overfitting shocks |
| 2026-03-05 | Outcome-aware weighting | Added `outcomes` capture + outcome multipliers | Policy now biased toward trajectories that lead to wins |
| 2026-03-05 | Augmentation and hardening | D4 symmetry + UI data logging fixes + tactical inference guards | Top-k and tactical behavior became significantly cleaner in play |

---

## 12) Dataset schema

Saved `.npz` files may contain:

- `states`: `(N, 3, 9, 9)` float32
- `actions`: `(N,)` int64
- `policy_targets` (optional): `(N, 81)` float32, row-normalized
- `outcomes` (optional): `(N,)` int8 in `{1,0,-1}`
- `current_players` (optional): `(N,)` int8 in `{1,-1}`

---

## 13) Tests

```bash
python -m unittest discover -s tests -v
```

Current tests validate:

- rule correctness and direction counting,
- game logging integrity,
- outcome weighting math and fallback behavior.

---

## 14) Limitations and next technical work

- Policy-only training: no value head yet.
- No search/rollout module (MCTS) yet.
- Tactic helper branches are heuristic gates, not learned tactical value heads.
- No distributed training in this baseline.

Recommended upgrades:

1. Policy+value multi-task training (auxiliary win-probability head).
2. Iterative teacher bootstrap loop (fresh self-play rounds after each checkpoint).
3. MCTS-guided policy improvement and stronger data curation.
4. Board-size generalization (13x13/15x15) with size-agnostic modules.
5. Confidence-calibrated inference (`softmax temperature`, entropy regularization, fallback smoothing).
