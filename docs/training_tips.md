# Training Command Guide

This document gives a practical command pipeline for dataset generation, iterative training, and evaluation.

## 1) Environment and prerequisites

From repository root:

```bash
python -m pip install -r requirements.txt
```

Confirm GPU/CPU availability:

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_device_count', torch.cuda.device_count())
PY
```

## 2) Data generation (fast baseline)

### 2.1 Heuristic self-play

```bash
python -m src.data.selfplay_generate --num-games 500 --output data/selfplay_500_heuristic.npz --seed 42 --noise 0.0
```

### 2.2 Soft labels with a teacher model

```bash
python -m src.data.selfplay_generate \
  --num-games 1000 \
  --seed 42 \
  --output data/selfplay_teacher_1k_soft.npz \
  --noise 0.05 \
  --teacher-checkpoint checkpoints/policy_net.pt \
  --teacher-device cuda \
  --teacher-temperature 1.0 \
  --teacher-hard-mix 0.15 \
  --save-soft-targets \
  --teacher-noise 0.02
```

Notes:
- Use `--teacher-noise` to avoid deterministic collapse.
- Use `--teacher-hard-mix` as a DAgger-style floor for conservative exploration.

## 3) Merge and inspect datasets

```bash
python - <<'PY'
import numpy as np
for p in [
    'data/selfplay_teacher_1k_soft.npz',
    'data/human_games.npz',
]:
    d = np.load(p)
    print(p, 'states', d['states'].shape, 'actions', d['actions'].shape)
    if 'policy_targets' in d:
        print(' soft-target', d['policy_targets'].shape)
    if 'outcomes' in d:
        from collections import Counter
        print(' outcomes', dict(Counter(d['outcomes'].tolist())))
PY
```

## 4) Canonical training stage (baseline strong setup)

```bash
python -m src.training.train_supervised \
  --data data/selfplay_teacher_1k_soft.npz \
  --extra-data data/human_games.npz \
  --target-mode auto \
  --epochs 120 \
  --batch-size 128 \
  --lr 0.0003 \
  --device cuda \
  --save-path checkpoints/policy_net.pt \
  --symmetry-augment \
  --split-path checkpoints/train_val_split_outcome.npz \
  --outcome-win-weight 1.5 \
  --outcome-loss-weight 1.2 \
  --outcome-draw-weight 1.0 \
  --defense-weight 3.0 \
  --attack-weight 3.0 \
  --threat-weight 1.8 \
  --build-threat-weight 2.0 \
  --threat-threshold 2000 \
  --weight-scan-radius 2 \
  --weight-progress-every 1000 \
  --resume-from checkpoints/policy_net.pt
```

## 5) Strong improvement loop (self-improving)

1. Generate `N` model-teacher games from current checkpoint.
2. Train with mixed sources (`--data` + `--extra-data`).
3. Rotate checkpoint and repeat.

Example loop:

```bash
for i in 0 1 2 3 4; do
  python -m src.data.selfplay_generate --num-games 5000 --output data/selfplay_iter_${i}.npz --teacher-checkpoint checkpoints/policy_net.pt --teacher-device cuda --seed $((100+i)) --save-soft-targets --teacher-temperature 1.0 --teacher-hard-mix 0.10 --noise 0.02
  python -m src.training.train_supervised --data data/selfplay_iter_${i}.npz --extra-data data/selfplay_teacher_1k_soft.npz --target-mode auto --epochs 80 --batch-size 128 --lr 0.0003 --device cuda --save-path checkpoints/policy_net_iter_${i}.pt --symmetry-augment --split-path checkpoints/train_val_split_iter_${i}.npz --outcome-win-weight 1.5 --outcome-loss-weight 1.2 --outcome-draw-weight 1.0 --defense-weight 3.0 --attack-weight 3.0 --threat-weight 1.8 --build-threat-weight 2.0
 done
```

## 6) Quick play check

```bash
python main.py --mode human-vs-model --model-path checkpoints/policy_net.pt --human-player black --device cuda
```

GUI:

```bash
python main.py --ui gui --mode human-vs-model --model-path checkpoints/policy_net.pt --human-player black --device cuda
```

## 7) Val-loss recovery playbook

- If val-loss regresses after policy changes: lower tactical multipliers temporarily and increase sample diversity.
- If early stopping triggers too early: increase patience, reduce LR decay, or increase dataset size.
- If split mismatch occurs: delete split file and regenerate.
- If load errors happen when starting GUI/terminal play: verify checkpoint exists and matches current architecture.

## 8) High-signal flag presets

### Conservative tactical style
`--defense-weight 4.0 --attack-weight 2.5 --threat-weight 2.0 --build-threat-weight 1.8`

### Aggressive attack style
`--defense-weight 2.0 --attack-weight 4.0 --threat-weight 1.3 --build-threat-weight 2.5`

## 9) Minimal single-run recipe (quick smoke test)

```bash
python -m src.data.selfplay_generate --num-games 200 --output data/smoke.npz --seed 7 --noise 0.02
python -m src.training.train_supervised --data data/smoke.npz --epochs 20 --batch-size 64 --lr 0.0005 --device cuda --save-path checkpoints/policy_net_smoke.pt --symmetry-augment
python main.py --mode human-vs-model --model-path checkpoints/policy_net_smoke.pt --human-player black
```
