# PI-Net V2: capacity-first redesign

This branch addresses the fixed-seed finding that PI-Net lost basic mission
capacity before contested-information effects were even isolated.

## What changed

1. **Reactive evidence backbone**
   - The actor now has a direct feed-forward path from the complete current
     decentralized structured observation (self state, stable entity slots,
     evidence mask, evidence metadata) to the seven action logits.
   - This path does not use centralized truth or future simulator state.

2. **Explicit first-discovery/search state**
   - Unknown targets are not assigned fictitious coordinates.
   - The policy receives explicit summary features for target visibility,
     target-known fraction, unknown-target fraction, and known-target age.
   - A small search residual can therefore learn behaviour before a target is
     ever observed.

3. **PI reasoning becomes a residual**
   - Physical/information future reasoning is preserved.
   - Its contribution starts at about 0.10 of the logit residual and is
     learnable.
   - The search residual starts at about 0.30 and is learnable.
   - This lets PPO first acquire competent reactive control instead of forcing
     basic pursuit/avoidance to be learned through the full counterfactual
     lattice.

4. **Stronger PPO defaults**
   - PPO epochs: 4 (was effectively 1 in the trainer CLI).
   - Base learning rate: 8e-5 (was 4e-5).
   - Entropy coefficient: 0.005 (was 0.01).
   - Checkpoints identify themselves as pi_mappo_v2_capacity_first.

5. **Curriculum transfer support**
   - --warm-start-actors-only may intentionally move a V2 actor from
     reference to contested while resetting critic and optimizers.
   - Normal full-checkpoint resume still rejects scenario mismatch.

## Recommended first run

Do not resume V1 weights. Start V2 from scratch.

Stage 1 learns basic mission control on the reference scenario:

```bash
python scripts/train_pi_mappo_parallel.py \
  --episodes 4000 \
  --scenario reference \
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
  --output outputs/pi_v2_reference_4000
```

Stage 2 transfers only the V2 actors into the contested scenario and relearns
the centralized critic under the harder dynamics:

```bash
python scripts/train_pi_mappo_parallel.py \
  --episodes 12000 \
  --resume outputs/pi_v2_reference_4000/checkpoint_004000.pt \
  --warm-start-actors-only \
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
  --output outputs/pi_v2_contested_12000
```

The stage-2 final global episode number is 12000, so this adds 8000 contested
episodes after the 4000-episode reference curriculum.

## Intended ablations later

Once V2 has recovered basic task capacity, the clean ablations are:
- reactive backbone only;
- reactive + persistent belief;
- reactive + belief + physical future;
- full reactive + physical/information future + cognitive gate.

The immediate goal is not to claim improvement. It is to restore basic mission
capacity before evaluating the information-reasoning contribution.
