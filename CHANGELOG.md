# Changelog

All entries are ordered by date and grouped by pipeline evolution.

## 2026-03-03

### Initial supervised baseline and game loop wiring
- Set up initial 9x9 Gomoku environment, rules, heuristic policy, and policy network.
- Added first generation/training loop over heuristic self-play data.
- Added terminal and logging entry points.
- Baseline training objective: hard labels with CE on 81-way action policy.

### Notable run
- Command style: `self-play dataset generation + train_supervised` with small data (few hundred moves).
- Observed validation behavior: very high loss regime (`~4.12`) and weak tactical consistency.
- Interpretation: data sparsity + label weakness dominated signal.

## 2026-03-03 (later)

### Soft targets and model-data plumbing
- Added model-based teacher path to self-play generation.
- Added `--teacher-checkpoint`, `--teacher-device`, `--teacher-temperature`, `--teacher-noise`, `--teacher-hard-mix`.
- Added optional `policy_targets` output in `.npz` datasets.
- Training now supports `--target-mode auto|hard|soft` and auto-detects soft labels when available.

### Outcome-aware direction planning
- Added game outcome capture path (`current_players`, `outcomes`) for logged and generated games.
- Mapped outcome labels:
  - `1`: eventual winner moves
  - `-1`: eventual loser moves
  - `0`: draw
- Added outcome weighting CLI knobs:
  - `--outcome-win-weight`
  - `--outcome-loss-weight`
  - `--outcome-draw-weight`

## 2026-03-04

### Tactical sample reweighting
- Added threat utilities in preprocessing:
  - `immediate_winning_actions`
  - `max_player_threat_score`
  - thresholded threat scoring with `--threat-threshold`
- Added tactical multipliers:
  - `--attack-weight`
  - `--defense-weight`
  - `--threat-weight`
  - `--build-threat-weight`
  - `--weight-scan-radius`
  - `--weight-progress-every`
- Added candidate filtering by local radius during tactical scoring for performance.

### Quality-of-train improvements
- Added symmetry augmentation toggle (`--symmetry-augment`, `--no-symmetry-augment`) via D4 transforms for states and action indices.
- Added `--split-path` persistence for stable train/validation indices.
- Added `--resume-from` checkpoint resume semantics.

### Notable run
- Command style with weighted tactics and soft targets:
  - `train_supervised ... --epochs 60 --batch-size 128 --lr 0.0005 --symmetry-augment`
- Significant practical improvement relative to initial regime; competitive policy behavior appeared despite still unstable tactical misses in some branches.

## 2026-03-05

### Outcome weighting operationalization
- Implemented outcome-aware weighted loss in training and validation numerics.
- Added validation/sample weighting consistency and updated metric accounting.
- Added tests validating outcome weight math.

### Notable run (full training block)
- Human-in-the-loop and self-play mixed datasets were used with:
  - `--weight-progress-every 1000`
  - `--symmetry-augment`
  - outcome weights and tactical weights enabled
- Example outcome: validation loss entered sub-1.x phase in long runs, with best checkpoints saved near `0.94` in one run.
- Added `early-stop` and LR scheduler tuning to stabilize later phases.

### Inference robustness
- Added/expanded legality masking and tactical fallback branches in `ModelAgent`:
  - immediate tactical win
  - must-block responses
  - strong threat response filtering
  - local proximity and threat-aware scoring bonuses.

### UI behavior refinement
- Added GUI logging workflow messages and optional replay/quit prompt after game completion.
- Added support paths for model-vs-human in both terminal and popup modes with cleaner status reporting.

## 2026-03-05 (current)

### Current documented state
- Training pipeline supports:
  - hard labels and soft labels
  - outcome-weighted learning
  - tactical sample weights
  - symmetry augmentation
  - LR scheduling + early stopping
  - reproducible resume and fixed split control
- Data logging supports appending and outcome-aware session boundaries.
- README expanded as full technical project documentation.

### Ongoing recommended next step
- Add closed-loop policy iteration:
  1. Generate with latest checkpoint as teacher
  2. Retrain with mixed-source soft labels + outcome weights
  3. Repeat with fresh fixed split policy

## 2026-03-05 (docs)

### Technical documentation upgrade
- Rewrote `README.md` into a publication-style technical report with full ML math and implementation detail.
- Added explicit optimization formulas for hard/soft targets and sample weights.
- Added a chronological experiment ledger and decision logs with observed effects.
- Made trial history discoverable via `CHANGELOG.md` link for repository upload and reproducibility review.
