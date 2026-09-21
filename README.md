# info-mappo-clean

A clean MAPPO starting point for the current contested UAV environment.

Kept:
- current reference/contested environment;
- hidden jammer truth;
- self comm_quality / recon_quality;
- idealized subnet sharing;
- MAPPO actor/critic;
- one training script.

Training unit:
- 1 round = num_envs complete environment episodes collected in parallel + 1 PPO update.
- With num_envs=16, the model still performs 1 update round, not 16 update rounds.

Quick speed test:

```bash
.venv/bin/python scripts/train_mappo_clean.py \
  --rounds 10 \
  --num-envs 16 \
  --device cuda \
  --log-every 1 \
  --save-every 10 \
  --output outputs/mappo_speed10
```

Normal training prints every 100 rounds by default.
