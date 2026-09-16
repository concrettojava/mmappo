from __future__ import annotations
import math
from typing import List
import numpy as np
from .entities import UAV, Target, Threat
from .scenario import reset_scene
from . import dynamics, communication, combat, observation, state, reward


SCENARIOS = ("reference", "contested")


class CooperativeUAVEnv:
    """Scene-level reproduction plus an optional contested-information scenario.

    ``reference`` preserves the reproduced Wang et al. scenario.

    ``contested`` keeps the same mission, reward and action space but adds
    spatially and temporally varying electromagnetic interference plus evasive
    target motion.  The interference degrades communication and reconnaissance
    ranges without exposing jammer state to the baseline policy.  This creates
    intermittent local observations/topology fragmentation while preserving the
    same fixed-vector dimensions, so the existing MAPPO baseline is a fair
    capacity probe rather than a new algorithm.
    """

    def __init__(
        self,
        seed: int = 0,
        world_size: float = 4000.0,
        dt: float = 1.0,
        max_steps: int = 200,
        target_speed: float = 8.0,
        target_turn_accel_std_deg_s2: float = 0.6,
        threat_eta: float = 1.8,
        paper_equation_yaw: bool = False,
        scenario: str = "reference",
        jammer_count: int = 2,
        jammer_min_radius: float = 700.0,
        jammer_max_radius: float = 1000.0,
        jammer_min_strength: float = 0.55,
        jammer_max_strength: float = 0.75,
        contested_min_comm_factor: float = 0.30,
        contested_min_recon_factor: float = 0.55,
        evasive_trigger_range: float = 700.0,
        evasive_turn_rate_deg_s: float = 10.0,
    ):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario={scenario!r}; expected one of {SCENARIOS}")
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.world_size = float(world_size)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.target_speed = float(target_speed)
        self.target_turn_accel_std = math.radians(target_turn_accel_std_deg_s2)
        self.threat_eta = float(threat_eta)
        self.paper_equation_yaw = paper_equation_yaw
        self.scenario = scenario

        self.jammer_count = int(jammer_count)
        self.jammer_min_radius = float(jammer_min_radius)
        self.jammer_max_radius = float(jammer_max_radius)
        self.jammer_min_strength = float(jammer_min_strength)
        self.jammer_max_strength = float(jammer_max_strength)
        self.contested_min_comm_factor = float(contested_min_comm_factor)
        self.contested_min_recon_factor = float(contested_min_recon_factor)
        self.evasive_trigger_range = float(evasive_trigger_range)
        self.evasive_turn_rate = math.radians(evasive_turn_rate_deg_s)

        self.uavs: List[UAV] = []
        self.targets: List[Target] = []
        self.threats: List[Threat] = []
        self.jammers: list[dict] = []
        self.step_count = 0
        self._geometry_cache = None

    @property
    def contested(self) -> bool:
        return self.scenario == "contested"

    def jammer_intensity_at(self, x: float, y: float) -> float:
        """Return deterministic aggregate interference intensity in [0, 1]."""
        if not self.contested or not self.jammers:
            return 0.0
        # Different jammer periods/phases create topology changes even for a
        # stationary formation.  No RNG is consumed here, so repeated graph/
        # observation calls within a step are exactly consistent.
        total_survival = 1.0
        for j in self.jammers:
            dx = float(x) - j["x"]
            dy = float(y) - j["y"]
            d2 = dx * dx + dy * dy
            spatial = math.exp(-d2 / max(j["radius"] * j["radius"], 1.0))
            temporal = 0.65 + 0.35 * math.sin(
                2.0 * math.pi * self.step_count / j["period"] + j["phase"]
            )
            local = float(np.clip(j["strength"] * spatial * temporal, 0.0, 0.95))
            total_survival *= 1.0 - local
        return float(np.clip(1.0 - total_survival, 0.0, 0.95))

    def communication_factor(self, uav: UAV) -> float:
        if not self.contested:
            return 1.0
        intensity = self.jammer_intensity_at(uav.x, uav.y)
        return float(max(self.contested_min_comm_factor, 1.0 - intensity))

    def reconnaissance_factor(self, uav: UAV) -> float:
        if not self.contested:
            return 1.0
        intensity = self.jammer_intensity_at(uav.x, uav.y)
        # Sensing degrades more gently than communication; otherwise the task
        # becomes an artificial blackout rather than an information-freshness
        # problem.
        return float(max(self.contested_min_recon_factor, 1.0 - 0.60 * intensity))

    def reset(self, seed: int | None = None):
        self._geometry_cache = None
        reset_scene(self, seed)
        return self.get_observations(), self.get_global_state()

    def reset_vectors(self, vectorizer, seed: int | None = None):
        """Reset and return fixed vectors without structured observation/state objects."""
        self._geometry_cache = None
        reset_scene(self, seed)
        return vectorizer.encode_env(self)

    def _dist(self, a, b):
        cache = self._geometry_cache
        if cache is not None:
            if isinstance(a, UAV) and isinstance(b, UAV):
                return float(cache["uav_uav"][a.idx, b.idx])
            if isinstance(a, UAV) and isinstance(b, Target):
                return float(cache["uav_target"][a.idx, b.idx])
            if isinstance(a, Target) and isinstance(b, UAV):
                return float(cache["uav_target"][b.idx, a.idx])
            if isinstance(a, UAV) and isinstance(b, Threat):
                return float(cache["uav_threat"][a.idx, b.idx])
            if isinstance(a, Threat) and isinstance(b, UAV):
                return float(cache["uav_threat"][b.idx, a.idx])
        return dynamics._dist(a, b)

    @staticmethod
    def _wrap_angle(a):
        return dynamics._wrap_angle(a)

    def _advance_xy(self, x, y, yaw, speed):
        return dynamics._advance_xy(self, x, y, yaw, speed)

    def _update_uavs(self, action_indices):
        return dynamics._update_uavs(self, action_indices)

    def _update_targets(self):
        return dynamics._update_targets(self)

    def communication_graph(self):
        return communication.communication_graph(self)

    def communication_components(self, graph=None):
        return communication.communication_components(self, graph=graph)

    def _direct_detected_targets(self, uav):
        return communication._direct_detected_targets(self, uav)

    def _direct_detected_threats(self, uav):
        return communication._direct_detected_threats(self, uav)

    def shared_detection(self, **kwargs):
        return communication.shared_detection(self, **kwargs)

    def _bearing_error(self, uav, target):
        return combat._bearing_error(self, uav, target)

    def _automatic_strikes(self, shared):
        return combat._automatic_strikes(self, shared)

    def _apply_collisions(self):
        return combat._apply_collisions(self)

    def _apply_threat_damage(self):
        return combat._apply_threat_damage(self)

    def get_observations(self):
        return observation.get_observations(self)

    def get_global_state(self):
        return state.get_global_states(self)

    def prepare_step(self, action_indices):
        """Apply movement only, returning context needed to finish the step."""
        if len(action_indices) != len(self.uavs):
            raise ValueError(f"Expected {len(self.uavs)} actions, got {len(action_indices)}")
        previous_action_u = {u.idx: float(u.last_action_u) for u in self.uavs}
        self._geometry_cache = None
        self.step_count += 1
        self._update_uavs(action_indices)
        self._update_targets()
        return previous_action_u

    def _resolve_step(self, previous_action_u):
        """Resolve detection/combat/reward and return rewards, done and info."""
        reward_ctx = reward.build_reward_context(self, previous_action_u)
        shared = {
            uid: (
                set(reward_ctx.shared_targets.get(uid, set())),
                set(reward_ctx.shared_threats.get(uid, set())),
            )
            for uid in reward_ctx.active_before
        }

        strikes = self._automatic_strikes(shared)
        collision_dead = self._apply_collisions()
        threat_dead = self._apply_threat_damage()
        newly_destroyed = set(collision_dead) | set(threat_dead)

        rewards, reward_breakdown = reward.compute_rewards(self, reward_ctx, newly_destroyed)
        done = (
            self.step_count >= self.max_steps
            or all(not t.alive for t in self.targets)
            or all(not u.alive for u in self.uavs)
        )

        alive_uavs = sum(u.alive for u in self.uavs)
        alive_targets = sum(t.alive for t in self.targets)
        info = {
            "step": self.step_count,
            "strikes": strikes,
            "collision_dead": sorted(collision_dead),
            "threat_dead": sorted(threat_dead),
            "alive_uavs": alive_uavs,
            "alive_targets": alive_targets,
            "completion_ratio": 1.0 - alive_targets / len(self.targets),
            "survival_ratio": alive_uavs / len(self.uavs),
            "reward_breakdown": reward_breakdown,
            "scenario": self.scenario,
        }
        return rewards, done, info

    def finish_step(self, previous_action_u):
        """Reference path: resolve step then materialize structured outputs."""
        rewards, done, info = self._resolve_step(previous_action_u)
        return self.get_observations(), self.get_global_state(), rewards, done, info

    def finish_step_vectors(self, previous_action_u, vectorizer):
        """Training path: resolve step then directly emit fixed arrays."""
        rewards, done, info = self._resolve_step(previous_action_u)
        obs_vec, state_vec, active = vectorizer.encode_env(self)
        return obs_vec, state_vec, active, rewards, done, info

    def step(self, action_indices):
        previous_action_u = self.prepare_step(action_indices)
        return self.finish_step(previous_action_u)

    def step_vectors(self, action_indices, vectorizer):
        previous_action_u = self.prepare_step(action_indices)
        return self.finish_step_vectors(previous_action_u, vectorizer)
