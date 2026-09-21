# Scenario 2: Dynamic Contested Reconnaissance-Strike

## Research question

Can a decentralized heterogeneous UAV swarm maintain reconnaissance-strike performance when battlefield information becomes intermittently unavailable because communication/sensing quality changes over space and time, while targets reactively maneuver?

The goal is not to make the baseline fail artificially. Scenario 2 preserves the same battlefield, UAV types, action space, reward, target/threat counts and episode horizon as the validated reference scenario, and only introduces two realistic sources of partial observability that motivate predictive belief reasoning.

## Changes relative to the validated reference scenario

### 1. Spatial-temporal electromagnetic interference

Each episode contains two latent jammer fields. A jammer has a random center, effective radius, strength, temporal period and phase. The jammer state is not supplied to the baseline policy.

At position p and step t, jammer influence is a smooth Gaussian spatial field multiplied by a periodic temporal factor. Multiple jammer effects are combined probabilistically and clipped to [0, 0.95].

Communication range of each UAV is multiplied by

`max(0.30, 1 - interference)`.

Reconnaissance range is degraded more gently:

`max(0.55, 1 - 0.60 * interference)`.

This produces dynamic communication-component fragmentation and intermittent target visibility without a binary, hand-crafted blackout.

Default jammer parameters:

- count: 2
- radius: 700--1000 m
- strength: 0.55--0.75
- temporal period: 45--85 s
- random phase per episode
- communication floor: 30% of nominal range
- reconnaissance floor: 55% of nominal range

### 2. Reactive target maneuver

Targets keep the reference stochastic turn dynamics. In Scenario 2, when an alive UAV enters a 700 m trigger radius, the target additionally performs a bounded turn away from the nearest UAV. The maximum reactive turn rate is 10 deg/s.

This keeps the target physically slow (the reference target speed is unchanged) but makes old observations less reliable than in a pure random-walk target model.

## What is intentionally unchanged

- 4 km x 4 km battlefield
- 8 heterogeneous RSUAVs
- 4 mobile targets
- 3 threat areas
- UAV physical capability parameters
- 7 discrete steering actions
- 200-step horizon
- collision/threat/strike mechanics
- reward function and coefficients
- MAPPO architecture and hyperparameters
- fixed-vector dimensions (obs=184, state=280). The two new observation
  entries are `comm_quality` and `recon_quality` in the observing UAV's
  `self` record only; jammer truth and teammate quality measurements remain
  hidden.

Keeping these constant isolates the effect of information interruption and target nonstationarity.

## Why this scenario is relevant

Recent literature increasingly treats communication fragmentation, jamming, information timeliness (Age of Information), and partial observation as first-class multi-UAV coordination problems. Examples include:

- *Autonomous decision-making method for integrated reconnaissance and strike operations under local observation and limited communication* (2026), which studies communication-network fragmentation and time-varying battlefield entities with heterogeneous graph spatio-temporal reasoning.
- *Age of information minimization in distributed multi-UAV networks with short-packet communications via hierarchical multi-agent learning* (Chinese Journal of Aeronautics, 2026), which explicitly treats information freshness as a coordination objective.
- *Learning to reallocate: MAPPO-based spectrum and power optimization for UAV-UGV clusters with dynamic reconfiguration* (Computer Communications, 2026), which considers malicious jamming and dynamic network reconfiguration.
- *Jamming-resilient unmanned aerial vehicle reconnaissance strategy using hybrid ordinary differential equation and deep reinforcement learning* (Engineering Applications of Artificial Intelligence, 2026), which formulates jamming-resilient reconnaissance as a POMDP.

Scenario 2 is deliberately narrower than those full communication/networking problems: it creates the information discontinuity needed to study belief-state reasoning while keeping our cooperative reconnaissance-strike environment controlled.

## Baseline capacity-probe protocol

Train the existing fixed-vector feed-forward MAPPO from scratch for exactly 32,000 global episodes with the same training settings used for the validated reference baseline:

- num_envs = 128
- minibatch_size = 1024
- device = CUDA
- same MAPPO learning rate, GAE, clipping, entropy and PPO epochs
- no early stopping
- independent final deterministic actor-only evaluation on 32 seeds

The baseline experiment is a diagnostic, not the final comparison suite. The hypothesis is that the memoryless actor will degrade because current observations disappear/reappear and target behavior is less predictable from one frame.

## Success criterion for the scenario design

Scenario 2 is useful if all three conditions hold:

1. the task remains learnable (MAPPO improves substantially over its early-training behavior);
2. the converged MAPPO result is materially below the validated reference-scenario result;
3. failures correlate with information interruption rather than trivial impossibility (e.g. universal UAV destruction).

If MAPPO remains near the reference result, interference should be strengthened only after inspecting connectivity/visibility statistics. If MAPPO collapses almost completely, interference should be weakened before designing the new model.

## Intended method direction

The scenario is designed to support, but not bake in, a future method based on:

- entity-centric heterogeneous representation;
- predictive belief for temporarily unobserved targets/threats;
- confidence/uncertainty attached to predicted entity states;
- uncertainty-aware relational aggregation and, optionally, selective communication.

The new model should be evaluated on the same Scenario 2 generator with identical seeds and environment mechanics.
