from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set
import math
import numpy as np

ACTION_VALUES = np.array([-1.0, -0.4, -0.1, 0.0, 0.1, 0.4, 1.0], dtype=np.float32)

TYPE_PARAMS = {
    "Stk": dict(strike_range=280.0, strike_angle_deg=45.0, recon_range=320.0,
                comm_range=550.0, speed=40.0, max_turn_deg_s=36.0, collision_range=10.0),
    "Rec": dict(strike_range=150.0, strike_angle_deg=30.0, recon_range=850.0,
                comm_range=550.0, speed=36.0, max_turn_deg_s=18.0, collision_range=10.0),
    "Com": dict(strike_range=180.0, strike_angle_deg=36.0, recon_range=300.0,
                comm_range=1200.0, speed=38.0, max_turn_deg_s=24.0, collision_range=10.0),
}

@dataclass
class UAV:
    idx: int
    x: float
    y: float
    uav_type: str
    yaw: float = 0.0
    alive: bool = True
    threat_memory: Set[int] = field(default_factory=set)
    last_action_u: float = 0.0

    @property
    def p(self):
        return TYPE_PARAMS[self.uav_type]

@dataclass
class Target:
    idx: int
    x: float
    y: float
    yaw: float
    omega: float = 0.0
    alive: bool = True

@dataclass
class Threat:
    idx: int
    x: float
    y: float
    radius: float
