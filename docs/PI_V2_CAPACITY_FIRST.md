# PI-Net V2: capacity-first redesign

This branch addresses the fixed-seed finding that PI-Net lost basic mission
capacity before contested-information effects were even isolated.

## What changed

1. **Reactive evidence backbone**
   - The actor now has a direct feed-forward path from the complete current
     decentralized structured observation (self state, stable entity slots,
     evidence mask, evidence metadata) to the seven action logits.
   - V2.1 strengthens this path to two hidden layers of width 256, matching the
     proven MAPPO baseline actor depth so the "capacity-first" route is not a
     weaker representational bottleneck than the baseline.
   - This path does not use centralized truth or future simulator state.

2. **Explicit first-discovery/search state**
   - Unknown targets are not assigned fictitious coordinates.
   - The policy receives explicit summary features for target visibility,
     target-known fraction, unknown-target fraction, and known-target age.
   - A small search residual can therefore learn behaviour before a target is
     ever observed.

3. **PI reasoning becomes a residual**
   - Physical/information future reasoning is preserved.
   - V2.1 uses a ReZero-style learnable gate: the PI contribution starts at
     exactly zero, so untrained counterfactual reasoning cannot perturb the
     direct policy at initialization. The scalar gate itself receives gradient
     immediately and admits the PI branch as training finds it useful.
   - The search residual still starts at about 0.30 and is learnable; its final
     action layer is small-initialized.
   - This makes the capacity-first claim explicit: PPO can first acquire basic
     pursuit/avoidance/task control through the direct path, then add PI
     reasoning rather than learning both simultaneously from noisy logits.

4. **Stronger PPO defaults**
   - PPO epochs: 4 (was effectively 1 in the trainer CLI).
   - Base learning rate: 8e-5 (was 4e-5).
   - Entropy coefficient: 0.005 (was 0.01).
   - V2.1 checkpoints identify themselves as pi_mappo_v2_1_capacity_first.
   - Training logs expose the torch.compile backend plus PI/search residual
     scales so the long run shows whether the reasoning branch actually opens.

5. **Curriculum transfer support**
   - --warm-start-actors-only may intentionally move a V2 actor from
     reference to contested while resetting critic and optimizers.
   - Normal full-checkpoint resume still rejects scenario mismatch.

## Recommended first run

Do not resume V1 weights. Start V2 from scratch directly on the contested
scenario so the main comparison is not confounded by curriculum pretraining.

```bash
python scripts/train_pi_mappo_parallel.py \
  --episodes 8000 \
  --scenario contested \
  --reward-profile paper \
  --num-envs 16 \
  --device cuda \
  --ppo-epochs 4 \
  --sequence-env-minibatch-size 8 \
  --actor-learning-rate 8e-5 \
  --critic-learning-rate 1e-4 \
  --entropy-coef 0.005 \
  --batch-actor-updates \
  --actor-batch-size 8 \
  --save-every 500 \
  --log-every 16 \
  --output outputs/pi_v2_contested_8000
```

Keep the original paper reward for this restart. The previous task_aligned
profile is intentionally not used because its large early-death penalty and
weak cumulative time pressure pushed learning toward survival rather than
mission completion.

If direct contested training still cannot establish basic task capacity, the
trainer also supports an explicit reference-to-contested actor curriculum using
--warm-start-actors-only. That is a fallback training strategy, not the primary
run.

## Intended ablations later

Once V2 has recovered basic task capacity, the clean ablations are:
- reactive backbone only;
- reactive + persistent belief;
- reactive + belief + physical future;
- full reactive + physical/information future + cognitive gate.

The immediate goal is not to claim improvement. It is to restore basic mission
capacity before evaluating the information-reasoning contribution.
