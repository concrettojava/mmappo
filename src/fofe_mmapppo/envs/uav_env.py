from __future__ import annotations
import math
import numpy as np
from typing import List
from .entities import UAV, Target, Threat
from .scenario import reset_scene
from . import dynamics, communication, combat, observation, state

class CooperativeUAVEnv:
    """
    Scene-level reproduction of Wang et al., FOFE-MMAPPO paper.

    Scope:
      - 4 km x 4 km battlefield
      - 8 heterogeneous RSUAVs
      - 4 moving targets
      - 3 fixed threat areas
      - communication topology + multi-hop subgroups
      - local detection + observation sharing
      - automatic strike
      - collision destruction
      - probabilistic threat destruction
      - 1 s step, 200-step episode

    This file intentionally does NOT implement MAPPO/FOFE/Mamba yet.

    Coordinate convention:
      The paper's Table 2 says yaw=0 means Heading North, but Eq.(5) writes
      xdot=v*cos(yaw), ydot=v*sin(yaw), which would mean yaw=0 points East.
      To match Fig.7 / "enter from the south", this reproduction uses:
          xdot = v*sin(yaw), ydot = v*cos(yaw)
      so yaw=0 points North.
      Set paper_equation_yaw=True to instead follow Eq.(5) literally.
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

    def reset(self, seed: int | None = None):
        reset_scene(self, seed)
        return self.get_observations(), self.get_global_state()

    @staticmethod
    def _dist(a, b):
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

    def communication_components(self):
        return communication.communication_components(self)

    def _direct_detected_targets(self, uav):
        return communication._direct_detected_targets(self, uav)

    def _direct_detected_threats(self, uav):
        return communication._direct_detected_threats(self, uav)

    def shared_detection(self):
        return communication.shared_detection(self)

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

    def step(self, action_indices):
        if len(action_indices) != len(self.uavs):
            raise ValueError(f"Expected {len(self.uavs)} actions, got {len(action_indices)}")

        self.step_count += 1
        self._update_uavs(action_indices)
        self._update_targets()

        shared = self.shared_detection()
        strikes = self._automatic_strikes(shared)
        collision_dead = self._apply_collisions()
        threat_dead = self._apply_threat_damage()

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
        }
        return self.get_observations(), self.get_global_state(), done, info
