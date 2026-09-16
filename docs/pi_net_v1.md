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
2. If evidence exists, first time-align stale position evidence to the current
   step using only its age, recorded heading and known entity speed, then
   perform evidence competition and gated correction.
3. Roll every entity hypothesis H steps into the physical future.
4. Analytically roll all seven UAV steering actions into H-step action
   primitives using known fixed-wing kinematics.
5. For every action/entity/hypothesis/future step, predict future observation
   and communication refresh probabilities.  The K futures remain separate
   until after those probabilities are predicted; the model does not first
   average multiple possible target positions into a fictitious mean target.
6. Aggregate hypothesis-conditioned refresh probabilities with mode weights,
   then propagate action-conditioned expected information age and uncertainty.
7. Build a physical-information interaction lattice with shape
   `[B, 7, 14, K, H, relation_dim]`.  Each lattice cell receives its own
   hypothesis-conditioned refresh signal.
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

## Stateful PPO constraint

PI-Net is stateful.  It must **not** be connected to the existing shuffled
feed-forward PPO minibatch update as if it were an MLP.  Doing so would break
temporal semantics.

`unroll_pi_actor` provides the first sequence-safe replay primitive.  For an
unchanged policy, replaying the same observation sequence from the same initial
belief must reproduce the rollout logits and log-probabilities, so the initial
PPO importance ratio is one.  Inactive synchronized-rollout padding is frozen,
and an explicit reset mask clears belief at episode boundaries.

The next engineering stage will build the recurrent PPO buffer/update path on
this sequence primitive rather than flattening time.

## Validation gates before trainer integration

`tests/test_pi_net.py` checks:

- stable entity identity slots
- optional age/source metadata preservation
- forward tensor contracts
- finite logits and probabilities
- age growth when evidence is missing
- age reset when evidence reappears
- stale-evidence dead reckoning before physical correction
- hypothesis-conditioned information-future aggregation
- fixed-wing action primitive geometry
- differentiability/backpropagation
- selective recurrent-state reset at episode boundaries

`tests/test_pi_sequence.py` checks:

- rollout/replay logits and log-probability equality for an unchanged policy
- initial PPO importance ratio equal to one
- inactive padding does not advance belief state
- episode reset does not leak previous-episode memory
- gradients propagate through the sequence replay path

Only after these gates pass should the model be connected to recurrent PPO
training and then to longer experiments.
