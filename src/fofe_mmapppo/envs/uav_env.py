from __future__ import annotations
import math
import numpy as np
from typing import List
from .entities import UAV, Target, Threat
from .scenario import reset_scene
from . import dynamics, communication, combat, observation, state, reward

class CooperativeUAVEnv:
    """Scene-level reproduction of Wang et al., FOFE-MMAPPO paper."""

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
    ):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.world_size = float(world_size)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.target_speed = float(target_speed)
        self.target_turn_accel_std = math.radians(target_turn_accel_std_deg_s2)
        self.threat_eta = float(threat_eta)
        self.paper_equation_yaw = paper_equation_yaw

        self.uavs: List[UAV] = []
        self.targets: List[Target] = []
        self.threats: List[Threat] = []
        self.step_count = 0
        self._geometry_cache = None

    def reset(self, seed: int | None = None):
        self._geometry_cache = None
        reset_scene(self, seed)
        return self.get_observations(), self.get_global_state()

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
        """Apply movement only, returning context needed to finish the step.

        Parallel training uses this split so all environments can move first,
        then one batched geometry calculation can be shared by reward/combat.
        ``step`` still uses the same two phases for reference behavior.
        """
        if len(action_indices) != len(self.uavs):
            raise ValueError(f"Expected {len(self.uavs)} actions, got {len(action_indices)}")
        previous_action_u = {u.idx: float(u.last_action_u) for u in self.uavs}
        self._geometry_cache = None
        self.step_count += 1
        self._update_uavs(action_indices)
        self._update_targets()
        return previous_action_u

    def finish_step(self, previous_action_u):
        """Resolve detection, combat, reward, termination and observations."""
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

        info = {
            "step": self.step_count,
            "strikes": strikes,
            "collision_dead": sorted(collision_dead),
            "threat_dead": sorted(threat_dead),
            "alive_uavs": sum(u.alive for u in self.uavs),
            "alive_targets": sum(t.alive for t in self.targets),
            "completion_ratio": 1.0 - sum(t.alive for t in self.targets) / len(self.targets),
            "survival_ratio": sum(u.alive for u in self.uavs) / len(self.uavs),
            "reward_breakdown": reward_breakdown,
        }
        return self.get_observations(), self.get_global_state(), rewards, done, info

    def step(self, action_indices):
        previous_action_u = self.prepare_step(action_indices)
        return self.finish_step(previous_action_u)
