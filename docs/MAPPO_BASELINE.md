# Fixed-vector MAPPO baseline

This stage reproduces the paper's traditional MAPPO comparison path before
FOFE or Mamba is introduced.

## Paper-matched settings

Published values used as defaults:

- discount factor `gamma = 0.95`
- GAE lambda `0.95`
- PPO clip epsilon `0.1`
- PPO epochs `15`
- actor/critic learning rate `4e-5`
- episode horizon `200`
- discrete action count `7`

The paper states that traditional MAPPO uses a fixed-size vectorized
observation instead of flexible observation, but it does **not** publish the
exact padding, normalization, MLP width, entropy coefficient, value-loss
coefficient, gradient clipping or mini-batch size. Those are therefore explicit
implementation choices rather than claimed paper facts.

Current implementation choices:

- deterministic object ordering by index;
- zero padding plus a presence bit per object slot;
- positions/ranges divided by battlefield size;
- angles divided by pi;
- UAV type one-hot encoded;
- two-layer tanh MLP, hidden width 256;
- entropy coefficient 0.01;
- value coefficient 0.5;
- max gradient norm 0.5;
- mini-batch size 64.

The environment remains structured/flexible. Vectorization happens only in
`models/vectorizer.py`, so FOFE can later consume the original structured
observation without changing the environment.

## Install with uv

CPU/default PyTorch resolver:

```bash
uv sync --extra rl
```

If you need a specific CUDA/PyTorch build, install the appropriate PyTorch build
for your machine and then install this project editable without replacing it.

## Smoke tests

```bash
.venv/bin/python -m unittest tests/test_mappo_baseline.py -v
.venv/bin/python -m unittest tests/test_reward.py -v
.venv/bin/python -m unittest tests/test_observation_state.py -v
.venv/bin/python tests/check_regression.py
```

## Short training smoke run

Do this before any long training:

```bash
.venv/bin/python scripts/train_mappo.py \
  --episodes 10 \
  --device cpu \
  --log-every 1 \
  --save-every 10
```

The script writes `metrics.jsonl` and checkpoints under `outputs/mappo/`, which
is ignored by Git.

## Paper-length training

After the smoke run and learning-curve sanity check:

```bash
.venv/bin/python scripts/train_mappo.py --episodes 32000 --device cuda
```

## Evaluation

```bash
.venv/bin/python scripts/evaluate_mappo.py \
  outputs/mappo/checkpoint_032000.pt \
  --episodes 100 \
  --device cuda
```

Evaluation reports mean/std of completion ratio, survival ratio, completion
steps and mean per-agent cumulative reward.
