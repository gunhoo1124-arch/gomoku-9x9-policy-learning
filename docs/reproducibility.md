# Reproducibility Protocol

This document defines what to persist and how to reproduce a training run end-to-end.

## 1) Versioning and immutable inputs

For every experiment, record:

- codebase commit hash (or snapshot timestamp),
- command line arguments,
- source dataset paths and their byte size,
- random seed,
- checkpoint path/version,
- `--split-path` file used for train/val partitioning,
- torch/cuda availability.

Recommended artifact names:
- `checkpoints/policy_net_<yyyymmdd>_<hhmm>.pt`
- `data/selfplay_<tag>.npz`
- `checkpoints/train_val_split_<tag>.npz`

## 2) Deterministic seeding

In this repository, determinism starts from command `--seed` in generators and trainer.

- Generator seeds all move shuffles and optional noise draw points.
- Trainer seeds dataset shuffles and augmentation order.

For strict reproducibility:

```bash
python -m src.training.train_supervised \
  --seed 42 \
  --split-path checkpoints/train_val_split_expA.npz \
  ...
```

Changing seed without regenerating split files changes fold composition and will alter val-loss.

## 3) Stable train/val split governance

`train_supervised` supports reusable splits via `--split-path`.

If split file metadata does not match dataset shape, the trainer raises:

```text
ValueError: Existing split size does not match current dataset size.
```

Action:

- delete the stale split:

```bash
Remove-Item checkpoints\train_val_split_outcome.npz
```

- rerun training to regenerate a split of correct size.

## 4) Dataset integrity checks

Before training, run a schema sanity check:

- `states.shape == (N, 3, 9, 9)`
- `actions.shape == (N,)` and all values in `[0, 80]`
- if `policy_targets` exists: shape `(N, 81)` and row sums > 0
- if `outcomes` exists: values in `{-1, 0, 1}`

Quick checker:

```bash
python - <<'PY'
import numpy as np
p = 'data/selfplay_teacher_1k_soft.npz'
d = np.load(p)
states, actions = d['states'], d['actions']
assert states.ndim == 4 and states.shape[1:] == (3, 9, 9)
assert actions.ndim == 1 and ((0 <= actions).all() and (actions < 81).all())
if 'policy_targets' in d:
    pt = d['policy_targets']
    assert pt.shape == (len(actions), 81)
    assert (pt.sum(axis=1) > 0).all()
if 'outcomes' in d:
    out = d['outcomes']
    assert set(map(int, set(np.unique(out)))).issubset({-1, 0, 1})
print('OK', p, len(actions))
PY
```

## 5) Reproducibility ledger

Each significant run should append one short record to `CHANGELOG.md` or an experiment note:

- command string,
- best `val_loss`,
- final `top1`, `top5`,
- epoch and stopping reason,
- data version (`--data` and `--extra-data`),
- active flags (especially tactical/outcome weights and split path).

## 6) Checkpoint compatibility

`ModelAgent` supports several checkpoint structures. If load fails:

- check that file exists at runtime path (absolute/relative path issues are common on Windows shells),
- ensure saved format matches current `PolicyNet` architecture,
- retrain from a known-good base checkpoint if checkpoint metadata changed.

Use this load-safe pattern for compatibility:

- prefer state-dict checkpoints only,
- avoid mixing incompatible network heads,
- keep model evolution aligned with `src/models/policy_net.py`.

## 7) Command-line reproducibility template

Save and reuse the exact command for publication:

```bash
python -m src.training.train_supervised \
  --data data/selfplay_teacher_1k_soft.npz \
  --extra-data data/human_games.npz \
  --target-mode auto \
  --epochs 120 \
  --batch-size 128 \
  --lr 0.0003 \
  --device cuda \
  --seed 42 \
  --save-path checkpoints/policy_net_expA.pt \
  --split-path checkpoints/train_val_split_expA.npz \
  --symmetry-augment \
  --resume-from checkpoints/policy_net_base.pt \
  --outcome-win-weight 1.5 \
  --outcome-loss-weight 1.2 \
  --outcome-draw-weight 1.0 \
  --defense-weight 3.0 \
  --attack-weight 3.0 \
  --threat-weight 1.8 \
  --build-threat-weight 2.0 \
  --threat-threshold 2000 \
  --weight-scan-radius 2 \
  --weight-progress-every 1000
```

## 8) Reporting format

For each run report:

- best checkpoint path,
- `best val_loss`,
- epoch of best checkpoint,
- early-stop epoch,
- LR schedule milestones,
- final effective learning rate.

Keep this in CHANGELOG for cross-run comparisons.
