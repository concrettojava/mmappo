# Clean MAPPO line

This branch is the fresh starting point for the new information-aware model.

## What is kept

- Current `reference` and `contested` environments.
- Dynamic jammer effects on communication/reconnaissance.
- Idealized communication-subnet sharing semantics.
- Jammer truth remains hidden from the actor.
- `self.comm_quality` and `self.recon_quality` are observable.
- Fixed-vector actor observation: 184 dimensions.
- Centralized critic state: 280 dimensions.
- Original two-layer MLP MAPPO actor/critic.

No PI actor, action-conditioned future rollout, hypothesis lattice, recurrent PI
training, replay guard, profiling, or `torch.compile` machinery is part of this
line.

## Training progress semantics

Use:

```bash
python scripts/train_mappo_clean.py --rounds 10 --num-envs 16 --device cuda
```

The main progress unit is a **round/update**.

One round with `num_envs=16` means:

1. 16 independent environments each produce one complete episode trajectory.
2. Those 16 trajectories are collected into one rollout batch.
3. MAPPO performs one PPO update.

Therefore one round is **one policy-update round**, not one sampled episode.
The log keeps the main counter in rounds and separately reports `data_ep`, the
number of environment episodes collected.

By default the trainer prints once every 100 rounds and reports:

- current round mean return;
- mean return over the most recent 100 rounds;
- completion and survival;
- actor and critic losses;
- seconds per round;
- elapsed wall-clock time;
- ETA;
- estimated finish time.

## Suggested first runs

Quick smoke test:

```bash
python scripts/train_mappo_clean.py \
  --rounds 1 \
  --num-envs 1 \
  --ppo-epochs 1 \
  --minibatch-size 64 \
  --device cpu \
  --log-every 1 \
  --save-every 1 \
  --output outputs/smoke_clean
```

GPU speed check with the real default MAPPO update settings:

```bash
python scripts/train_mappo_clean.py \
  --rounds 10 \
  --num-envs 16 \
  --device cuda \
  --log-every 1 \
  --save-every 10 \
  --output outputs/mappo_speed10
```

Longer run:

```bash
python scripts/train_mappo_clean.py \
  --rounds 5000 \
  --num-envs 16 \
  --device cuda \
  --log-every 100 \
  --save-every 500 \
  --output outputs/mappo_clean_5000
```
