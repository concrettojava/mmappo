"""Reward functions from Wang et al., Eqs. (11)-(17).

This module keeps reward calculation separate from environment transition logic.
The implementation follows the paper as literally as possible and makes two
explicit engineering choices where the paper is operationally ambiguous:

1. Ability rewards are evaluated after movement/detection but before automatic
   strike/collision/threat destruction.  Otherwise the Eq. (14) strike
   effectiveness term |A_i^M| would always be zero because strikeable targets
   are immediately destroyed.
2. The -50 destroy penalty is applied once, on the transition where the UAV is
   newly destroyed.  Agents that were already dead before the transition get
   zero reward and take no further actions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping, Set


# Paper Section 5.1 coefficients.
LAMBDA_TIME = 1.0 / 100.0
LAMBDA_DIST = 1.0 / 2000.0
LAMBDA_MISSION = 1.0
LAMBDA_ABILITY = 1.0
LAMBDA_ACTION = 0.25
LAMBDA_BOUND = 1.0

ABILITY_RATIOS = {
    "Stk": {"Stk": 0.7, "Rec": 0.2, "Com": 0.1},
    "Rec": {"Stk": 0.1, "Rec": 0.7, "Com": 0.2},
    "Com": {"Stk": 0.2, "Rec": 0.1, "Com": 0.7},
}


@dataclass(frozen=True)
class RewardContext:
    """Pre-destruction information required by Eq. (13)-(15)."""

    active_before: Set[int]
    previous_action_u: Mapping[int, float]
    direct_targets: Mapping[int, Set[int]]
    direct_threats: Mapping[int, Set[int]]
    shared_targets: Mapping[int, Set[int]]
    shared_threats: Mapping[int, Set[int]]
    direct_neighbors: Mapping[int, Set[int]]
    strikeable_targets: Mapping[int, Set[int]]
    undiscovered_threats: Set[int]


def general_reward(a: float, b: float) -> float:
    """Eq. (11).

    The function is a bounded proximity/incompleteness penalty.  It is zero
    when ``a >= b`` (or ``b <= 0``), approximately -1 at ``a == 0``, and is
    clipped from below at -10 for negative ``a`` values such as boundary
    violations.
    """
    a = float(a)
    b = float(b)
    if b <= 0.0 or a >= b:
        return 0.0
    value = -(math.exp(-a / b) - math.exp(-1.0)) / (1.0 - math.exp(-1.0))
    return max(value, -10.0)


def _min_distance(env, uav, objects: Iterable[object]) -> float | None:
    distances = [env._dist(uav, obj) for obj in objects]
    return min(distances) if distances else None


def _strikeable_targets(env, uav, visible_target_ids: Set[int]) -> Set[int]:
    result = set()
    for tid in visible_target_ids:
        target = env.targets[tid]
        if not target.alive:
            continue
        distance = env._dist(uav, target)
        angle = abs(env._bearing_error(uav, target))
        if (
            distance < uav.p["strike_range"]
            and angle < math.radians(uav.p["strike_angle_deg"]) / 2.0
        ):
            result.add(tid)
    return result


def build_reward_context(env, previous_action_u: Mapping[int, float]) -> RewardContext:
    """Capture reward-relevant geometry after movement, before destruction."""
    active_before = {u.idx for u in env.uavs if u.alive}
    graph = env.communication_graph()

    direct_targets: Dict[int, Set[int]] = {}
    direct_threats: Dict[int, Set[int]] = {}
    for u in env.uavs:
        if not u.alive:
            continue
        direct_targets[u.idx] = env._direct_detected_targets(u)
        direct_threats[u.idx] = env._direct_detected_threats(u)

    # shared_detection also applies newly observed threat memory and uses the
    # current communication topology.
    shared = env.shared_detection()
    shared_targets = {uid: set(values[0]) for uid, values in shared.items()}
    shared_threats = {uid: set(values[1]) for uid, values in shared.items()}

    strikeable = {}
    for u in env.uavs:
        if u.alive:
            strikeable[u.idx] = _strikeable_targets(
                env, u, shared_targets.get(u.idx, set())
            )

    discovered = set().union(*(u.threat_memory for u in env.uavs)) if env.uavs else set()
    undiscovered = {th.idx for th in env.threats} - discovered

    return RewardContext(
        active_before=active_before,
        previous_action_u=dict(previous_action_u),
        direct_targets=direct_targets,
        direct_threats=direct_threats,
        shared_targets=shared_targets,
        shared_threats=shared_threats,
        direct_neighbors={uid: set(ns) for uid, ns in graph.items()},
        strikeable_targets=strikeable,
        undiscovered_threats=undiscovered,
    )


def mission_reward(env) -> float:
    """Eq. (12), evaluated on the post-transition mission status."""
    alive_targets = sum(t.alive for t in env.targets)
    alive_uavs = sum(u.alive for u in env.uavs)
    return (
        -LAMBDA_TIME * (env.step_count * env.dt)
        - general_reward(alive_targets, len(env.targets))
        + general_reward(alive_uavs, len(env.uavs))
    )


def avoidance_reward(env, uav, ctx: RewardContext) -> float:
    """R_i^Avoid in Eq. (13)."""
    value = 0.0
    for other in env.uavs:
        if other.idx == uav.idx or other.idx not in ctx.active_before:
            continue
        value += 5.0 * general_reward(env._dist(uav, other), 160.0)
    for threat in env.threats:
        value += 10.0 * general_reward(env._dist(uav, threat), 360.0)
    return value


def same_type_exclusion_reward(env, uav, radius_key: str, ctx: RewardContext) -> float:
    """The isomorphic-exclusion term used by all three Eq. (14) rewards."""
    threshold = 0.8 * uav.p[radius_key]
    value = 0.0
    for other in env.uavs:
        if (
            other.idx != uav.idx
            and other.idx in ctx.active_before
            and other.uav_type == uav.uav_type
        ):
            value += general_reward(env._dist(uav, other), threshold)
    return value


def strike_reward(env, uav, ctx: RewardContext) -> float:
    """R_i^Stk from Eq. (14)."""
    alive_targets = [t for t in env.targets if t.alive]
    nearest = _min_distance(env, uav, alive_targets)
    distance_term = -LAMBDA_DIST * nearest if nearest is not None else 0.0
    effectiveness = 10.0 * len(ctx.strikeable_targets.get(uav.idx, set()))
    exclusion = same_type_exclusion_reward(env, uav, "strike_range", ctx)
    return distance_term + effectiveness + exclusion


def reconnaissance_reward(env, uav, ctx: RewardContext) -> float:
    """R_i^Rec from Eq. (14).

    Target IDs and threat IDs belong to different mathematical sets in the
    paper.  Their integer IDs may overlap in code (e.g. target 0 and threat 0),
    so effectiveness is computed by summing the two cardinalities rather than
    taking a Python set union across namespaces.
    """
    alive_targets = [t for t in env.targets if t.alive]
    unknown_threats = [env.threats[k] for k in sorted(ctx.undiscovered_threats)]
    nearest = _min_distance(env, uav, [*alive_targets, *unknown_threats])
    distance_term = -LAMBDA_DIST * nearest if nearest is not None else 0.0

    known_target_ids = set(ctx.direct_targets.get(uav.idx, set()))
    known_threat_ids = (
        set(ctx.direct_threats.get(uav.idx, set()))
        | set(uav.threat_memory)
    )
    known_count = len(known_target_ids) + len(known_threat_ids)
    denominator = len(alive_targets) + len(env.threats)
    effectiveness = general_reward(known_count, denominator)
    exclusion = same_type_exclusion_reward(env, uav, "recon_range", ctx)
    return distance_term + effectiveness + exclusion


def communication_reward(env, uav, ctx: RewardContext) -> float:
    """R_i^Com from Eq. (14)."""
    others = [
        other for other in env.uavs
        if other.idx != uav.idx and other.idx in ctx.active_before
    ]
    nearest = _min_distance(env, uav, others)
    distance_term = -LAMBDA_DIST * nearest if nearest is not None else 0.0

    direct_count = len(ctx.direct_neighbors.get(uav.idx, set()))
    effectiveness = general_reward(direct_count, len(others))
    exclusion = same_type_exclusion_reward(env, uav, "comm_range", ctx)
    return distance_term + effectiveness + exclusion


def action_reward(uav, previous_u: float) -> float:
    """Eq. (15)."""
    current_u = float(uav.last_action_u)
    delta_u = current_u - float(previous_u)
    return -0.5 * abs(delta_u) - current_u * current_u


def boundary_reward(env, uav) -> float:
    """Eq. (16) for a [0, world_size] x [0, world_size] battlefield."""
    x_min = 0.0
    x_max = env.world_size
    y_min = 0.0
    y_max = env.world_size
    return (
        general_reward(uav.x - x_min, 160.0)
        + general_reward(x_max - uav.x, 160.0)
        + general_reward(uav.y - y_min, 160.0)
        + general_reward(y_max - uav.y, 160.0)
    )


def ability_reward(env, uav, ctx: RewardContext, newly_destroyed: Set[int]):
    """Eq. (13)-(14), returning the total and diagnostic components."""
    r_avoid = avoidance_reward(env, uav, ctx)
    r_destroy = -50.0 if uav.idx in newly_destroyed else 0.0
    r_stk = strike_reward(env, uav, ctx)
    r_rec = reconnaissance_reward(env, uav, ctx)
    r_com = communication_reward(env, uav, ctx)

    ratios = ABILITY_RATIOS[uav.uav_type]
    total = (
        r_avoid
        + r_destroy
        + ratios["Stk"] * r_stk
        + ratios["Rec"] * r_rec
        + ratios["Com"] * r_com
    )
    return total, {
        "avoid": r_avoid,
        "destroy": r_destroy,
        "strike": r_stk,
        "reconnaissance": r_rec,
        "communication": r_com,
    }


def compute_rewards(env, ctx: RewardContext, newly_destroyed: Set[int]):
    """Compute Eq. (17) for every agent plus a per-component breakdown."""
    r_mission = mission_reward(env)
    rewards: Dict[int, float] = {}
    breakdown: Dict[int, dict] = {}

    for uav in env.uavs:
        # No repeated reward after an agent has already left the process.
        if uav.idx not in ctx.active_before:
            rewards[uav.idx] = 0.0
            breakdown[uav.idx] = {
                "mission": 0.0,
                "ability": 0.0,
                "action": 0.0,
                "boundary": 0.0,
                "total": 0.0,
            }
            continue

        r_ability, ability_parts = ability_reward(env, uav, ctx, newly_destroyed)
        r_action = action_reward(uav, ctx.previous_action_u.get(uav.idx, 0.0))
        r_bound = boundary_reward(env, uav)

        total = (
            LAMBDA_MISSION * r_mission
            + LAMBDA_ABILITY * r_ability
            + LAMBDA_ACTION * r_action
            + LAMBDA_BOUND * r_bound
        )
        rewards[uav.idx] = float(total)
        breakdown[uav.idx] = {
            "mission": float(r_mission),
            "ability": float(r_ability),
            "action": float(r_action),
            "boundary": float(r_bound),
            "ability_parts": ability_parts,
            "total": float(total),
        }

    return rewards, breakdown
