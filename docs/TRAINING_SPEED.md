# MAPPO training speed path

This branch adds an engineering-optimized trainer while retaining
`scripts/train_mappo.py` as the single-environment reference path.

## Why the RTX 3080 was under-utilized

The reference trainer performs many tiny GPU operations:

- one environment at a time;
- eight independent actor/critic networks evaluated agent-by-agent;
- only ~200 transitions per agent before each update;
- mini-batch size 64;
- Python environment/vectorization work between GPU calls.

That pattern can show low SM utilization and low board power even when the
Windows Task Manager 3D graph looks high.

## Optimizations implemented

`scripts/train_mappo_parallel.py` adds:

1. **multi-environment rollout batches** via `--num-envs`;
2. **batched inference per agent**: agent `i` evaluates all active environments
   in one forward pass;
3. **parallel GAE bookkeeping** with no cross-environment leakage;
4. **larger PPO mini-batches** (default 1024);
5. **TF32-friendly float32 matmul precision** on supported NVIDIA GPUs;
6. **fused Adam on CUDA** when available;
7. **`torch.inference_mode()`** for rollout inference;
8. reduced metrics file I/O and exact checkpoint boundaries.

The environment transition itself is unchanged and still runs in Python.  This
is intentionally the first optimization stage: benchmark it before adding
multiprocessing, because process IPC can become a new bottleneck for 1-second
step simulations.

## Important reproducibility note

The paper mentions 32 independent random environments for training-process
indicator evaluation/statistics, but does not explicitly say that 32
environments are stepped together as a vectorized PPO rollout.  Therefore this
parallel trainer is an engineering acceleration, **not a claimed paper detail**.

Also, collecting several episodes before one PPO update changes optimization
dynamics relative to the single-environment reference implementation.  Keep the
single-environment results as the strict reproduction reference.

## Validate

```bash
.venv/bin/python -m unittest tests/test_parallel_training.py -v
.venv/bin/python -m unittest tests/test_mappo_baseline.py -v
.venv/bin/python -m unittest tests/test_reward.py -v
.venv/bin/python -m unittest tests/test_observation_state.py -v
.venv/bin/python tests/check_regression.py
```

## Short benchmark from an existing checkpoint

Do not interrupt a currently running experiment.  After a checkpoint is safely
written, benchmark into a separate output directory:

```bash
.venv/bin/python scripts/train_mappo_parallel.py \
  --episodes 700 \
  --resume outputs/mappo_500/checkpoint_000500.pt \
  --num-envs 16 \
  --minibatch-size 1024 \
  --device cuda \
  --log-every 32 \
  --save-every 100 \
  --output outputs/mappo_parallel_bench
```

The log prints `speed=... ep/s`.

Try `--num-envs 8`, `16`, and `32` in separate output folders.  The best value
is hardware/environment dependent.  More environments do not guarantee higher
speed because Python environment stepping eventually becomes the bottleneck.

## Continue a trained checkpoint

Example:

```bash
.venv/bin/python scripts/train_mappo_parallel.py \
  --episodes 5000 \
  --resume outputs/mappo_500/checkpoint_001000.pt \
  --num-envs 16 \
  --minibatch-size 1024 \
  --device cuda \
  --log-every 100 \
  --save-every 500 \
  --output outputs/mappo_fast
```

Older checkpoints without optimizer states still load model weights; Adam state
is restarted.  New checkpoints save both model and optimizer states.

## Possible second-stage optimizations

Only consider these after measuring the first-stage trainer:

- persistent multiprocessing environment workers;
- NumPy/vectorized distance, communication and reward calculations;
- tensor/GPU-resident rollout storage to reduce host-device copies;
- `torch.compile` after profiling (small MLPs do not always benefit);
- asynchronous environment reset/collection instead of waiting for the slowest
  environment in each rollout batch.
