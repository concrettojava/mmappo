# PI-Net V1 design contract

This branch introduces the first executable core of the proposed PI-Net actor.
It is intentionally **not yet wired into PPO training**.  The goal of this
stage is to validate the new computation structure before changing the rollout
buffer or trainer.

## Why a new tensorizer is required

The fixed-vector MAPPO baseline packs only currently available records.  That
is sufficient for a feed-forward baseline, but it is unsuitable for a
persistent per-entity belief: target/threat identity can be lost when records
disappear and later reappear.

`EntityTensorizer` therefore maps the decentralized structured observation to
stable slots:

- 7 teammate slots (global UAV identity, excluding self)
- 4 target slots
- 3 threat slots

No simulator-private information is added.

## PI actor state

Each actor maintains an explicit recurrent state for every external entity:

- K latent hypotheses
- hypothesis logits/probabilities
- physical position/yaw/speed/turn limit for every hypothesis
- information age
- known/alive masks
- a compact inferred information-environment context

The current V1 defaults are K=3 hypotheses and an 8-step future horizon.

## Forward computation

1. Predict each persistent entity belief forward by one real environment step.
2. If fresh evidence exists, perform evidence competition and gated correction.
3. Roll every entity hypothesis H steps into the physical future.
4. Analytically roll all seven UAV steering actions into H-step action
   primitives using known fixed-wing kinematics.
5. For every action/entity/future step, predict future observation and
   communication refresh probabilities from decentralized history/context.
6. Propagate action-conditioned expected information age and uncertainty.
7. Build a physical-information interaction lattice with shape
   `[B, 7, 14, K, H, relation_dim]`.
8. Read out task value and information value for each action, then combine them
   through a learned cognitive-demand gate.

## Execution-time information boundary

The actor input deliberately excludes:

- true jammer coordinates or jammer intensity
- hidden target truth
- global communication graph
- other agents' private observations
- future simulator states

Those variables may later be used only as training labels for auxiliary
prediction losses.

## Important training constraint

PI-Net is stateful.  It must **not** be connected to the existing shuffled
feed-forward PPO minibatch update as if it were an MLP.  Doing so would break
temporal semantics.  The next engineering stage will add a recurrent/sequence
PPO path (or explicitly stored belief-state inputs) and validate policy-ratio
consistency before long training runs.

## Validation gates before trainer integration

`tests/test_pi_net.py` checks:

- stable entity identity slots
- optional age/source metadata preservation
- forward tensor contracts
- finite logits and probabilities
- age growth when evidence is missing
- age reset when evidence reappears
- fixed-wing action primitive geometry
- differentiability/backpropagation
- selective recurrent-state reset at episode boundaries

Only after these tests pass should the model be connected to rollout/training.
