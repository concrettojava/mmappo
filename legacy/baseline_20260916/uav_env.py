
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
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

# Paper Table 2
INITIAL_UAVS = [
    # id, x, y, type
    (0, 1700.0, 200.0, "Stk"),
    (1, 1900.0, 200.0, "Stk"),
    (2, 2100.0, 200.0, "Stk"),
    (3, 2300.0, 200.0, "Stk"),
    (4, 1800.0, 300.0, "Rec"),
    (5, 2200.0, 300.0, "Rec"),
    (6, 1800.0, 100.0, "Com"),
    (7, 2200.0, 100.0, "Com"),
]

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
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.uavs = [UAV(idx=i, x=x, y=y, uav_type=t, yaw=0.0)
                     for i, x, y, t in INITIAL_UAVS]

        # Paper: targets and threat areas are randomly distributed in battlefield.
        # Use a margin only to avoid degenerate spawning on the exact boundary.
        margin = 250.0
        self.targets = []
        for j in range(4):
            x = float(self.rng.uniform(margin, self.world_size - margin))
            y = float(self.rng.uniform(1200.0, self.world_size - margin))
            yaw = float(self.rng.uniform(-math.pi, math.pi))
            self.targets.append(Target(j, x, y, yaw))

        self.threats = []
        for k in range(3):
            x = float(self.rng.uniform(margin, self.world_size - margin))
            y = float(self.rng.uniform(700.0, self.world_size - margin))
            radius = float(self.rng.uniform(120.0, 200.0))
            self.threats.append(Threat(k, x, y, radius))

        return self.get_observations(), self.get_global_state()

    @staticmethod
    def _dist(a, b) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _advance_xy(self, x, y, yaw, speed):
        if self.paper_equation_yaw:
            # Literal Eq.(5)/(6)
            x += speed * math.cos(yaw) * self.dt
            y += speed * math.sin(yaw) * self.dt
        else:
            # Match Table 2/Fig.7 convention: yaw=0 points North.
            x += speed * math.sin(yaw) * self.dt
            y += speed * math.cos(yaw) * self.dt
        return x, y

    def _update_uavs(self, action_indices):
        for uav, aidx in zip(self.uavs, action_indices):
            if not uav.alive:
                continue
            u = float(ACTION_VALUES[int(aidx)])
            uav.last_action_u = u
            max_turn = math.radians(uav.p["max_turn_deg_s"])
            omega = max_turn * u
            uav.yaw = self._wrap_angle(uav.yaw + omega * self.dt)
            uav.x, uav.y = self._advance_xy(uav.x, uav.y, uav.yaw, uav.p["speed"])

    def _update_targets(self):
        for t in self.targets:
            if not t.alive:
                continue
            alpha = float(self.rng.normal(0.0, self.target_turn_accel_std))
            t.omega += alpha * self.dt
            t.yaw = self._wrap_angle(t.yaw + t.omega * self.dt)
            nx, ny = self._advance_xy(t.x, t.y, t.yaw, self.target_speed)

            # Paper only states targets remain in mission area.
            # Reflect at boundaries as a minimally invasive implementation.
            if nx < 0 or nx > self.world_size:
                t.yaw = self._wrap_angle(-t.yaw)
            if ny < 0 or ny > self.world_size:
                t.yaw = self._wrap_angle(math.pi - t.yaw)
            t.x, t.y = self._advance_xy(t.x, t.y, t.yaw, self.target_speed)
            t.x = float(np.clip(t.x, 0.0, self.world_size))
            t.y = float(np.clip(t.y, 0.0, self.world_size))

    def communication_graph(self) -> Dict[int, Set[int]]:
        alive = [u for u in self.uavs if u.alive]
        graph = {u.idx: set() for u in alive}
        for i, a in enumerate(alive):
            for b in alive[i+1:]:
                # Paper Eq.(1): direct communication if distance is within
                # max(comm_range_i, comm_range_j).
                if self._dist(a, b) < max(a.p["comm_range"], b.p["comm_range"]):
                    graph[a.idx].add(b.idx)
                    graph[b.idx].add(a.idx)
        return graph

    def communication_components(self) -> Dict[int, Set[int]]:
        graph = self.communication_graph()
        comps = {}
        visited = set()
        for node in graph:
            if node in visited:
                continue
            stack = [node]
            comp = set()
            while stack:
                n = stack.pop()
                if n in comp:
                    continue
                comp.add(n)
                visited.add(n)
                stack.extend(graph[n] - comp)
            for n in comp:
                comps[n] = set(comp)
        return comps

    def _direct_detected_targets(self, uav: UAV) -> Set[int]:
        return {t.idx for t in self.targets
                if t.alive and self._dist(uav, t) < uav.p["recon_range"]}

    def _direct_detected_threats(self, uav: UAV) -> Set[int]:
        found = {th.idx for th in self.threats
                 if self._dist(uav, th) < uav.p["recon_range"]}
        uav.threat_memory |= found
        return found

    def shared_detection(self):
        comps = self.communication_components()
        direct_targets = {}
        direct_threats = {}
        for u in self.uavs:
            if u.alive:
                direct_targets[u.idx] = self._direct_detected_targets(u)
                direct_threats[u.idx] = self._direct_detected_threats(u)

        result = {}
        for u in self.uavs:
            if not u.alive:
                continue
            comp = comps.get(u.idx, {u.idx})
            visible_targets = set().union(*(direct_targets.get(v, set()) for v in comp))
            visible_threats = set().union(*(
                (direct_threats.get(v, set()) |
                 next(x for x in self.uavs if x.idx == v).threat_memory)
                for v in comp
            ))
            result[u.idx] = (visible_targets, visible_threats)
        return result

    def _bearing_error(self, uav: UAV, target: Target) -> float:
        dx, dy = target.x - uav.x, target.y - uav.y
        if self.paper_equation_yaw:
            bearing = math.atan2(dy, dx)
        else:
            # yaw=0 North
            bearing = math.atan2(dx, dy)
        return self._wrap_angle(bearing - uav.yaw)

    def _automatic_strikes(self, shared):
        destroyed = []
        for u in self.uavs:
            if not u.alive:
                continue
            visible_targets, _ = shared.get(u.idx, (set(), set()))
            for tid in sorted(visible_targets):
                t = self.targets[tid]
                if not t.alive:
                    continue
                d = self._dist(u, t)
                angle = abs(self._bearing_error(u, t))
                if (d < u.p["strike_range"] and
                    angle < math.radians(u.p["strike_angle_deg"]) / 2.0):
                    t.alive = False
                    destroyed.append((u.idx, tid))
        return destroyed

    def _apply_collisions(self):
        dead = set()
        alive = [u for u in self.uavs if u.alive]
        for i, a in enumerate(alive):
            for b in alive[i+1:]:
                if self._dist(a, b) < min(a.p["collision_range"], b.p["collision_range"]):
                    dead.add(a.idx)
                    dead.add(b.idx)
        for idx in dead:
            self.uavs[idx].alive = False
        return dead

    def _apply_threat_damage(self):
        dead = set()
        for u in self.uavs:
            if not u.alive:
                continue
            for th in self.threats:
                rho = self._dist(u, th)
                if rho >= th.radius:
                    continue
                # Paper Eq.(7), with DeltaT = dt.
                base = 1.0 - (1.0 - rho / th.radius) ** self.threat_eta
                p_shot = 1.0 - base ** self.dt
                if self.rng.random() < p_shot:
                    u.alive = False
                    dead.add(u.idx)
                    break
        return dead

    def get_observations(self):
        shared = self.shared_detection()
        comps = self.communication_components()
        observations = {}

        for u in self.uavs:
            if not u.alive:
                observations[u.idx] = None
                continue

            comp = comps.get(u.idx, {u.idx})
            neighbor_ids = sorted(comp - {u.idx})
            visible_targets, visible_threats = shared[u.idx]

            observations[u.idx] = {
                "self": {
                    "idx": u.idx, "type": u.uav_type, "alive": u.alive,
                    "x": u.x, "y": u.y, "yaw": u.yaw,
                },
                "neighbors": [
                    {
                        "idx": v, "type": self.uavs[v].uav_type,
                        "alive": self.uavs[v].alive,
                        "x": self.uavs[v].x, "y": self.uavs[v].y,
                        "yaw": self.uavs[v].yaw,
                    } for v in neighbor_ids
                ],
                "targets": [
                    {
                        "idx": tid, "alive": self.targets[tid].alive,
                        "x": self.targets[tid].x, "y": self.targets[tid].y,
                        "yaw": self.targets[tid].yaw,
                    } for tid in sorted(visible_targets)
                ],
                "threats": [
                    {
                        "idx": kid, "radius": self.threats[kid].radius,
                        "x": self.threats[kid].x, "y": self.threats[kid].y,
                    } for kid in sorted(visible_threats)
                ],
            }
        return observations

    def get_global_state(self):
        return {
            "uavs": [vars(u).copy() for u in self.uavs],
            "targets": [vars(t).copy() for t in self.targets],
            "threats": [vars(th).copy() for th in self.threats],
        }

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

    def render(self, ax=None, show_comm=True, show_ranges=False):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7))

        ax.clear()
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_aspect("equal")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.set_title(f"FOFE-MMAPPO scene reproduction — step {self.step_count}")

        # Threats
        for th in self.threats:
            circ = plt.Circle((th.x, th.y), th.radius, fill=False, linestyle="--")
            ax.add_patch(circ)
            ax.scatter([th.x], [th.y], marker="X", s=70)
            ax.text(th.x + 25, th.y + 25, f"T{th.idx}")

        # Targets
        for t in self.targets:
            if t.alive:
                ax.scatter([t.x], [t.y], marker="s", s=45)
                ax.text(t.x + 20, t.y + 20, f"M{t.idx}")

        marker = {"Stk": "v", "Rec": "D", "Com": "o"}
        for u in self.uavs:
            if not u.alive:
                ax.scatter([u.x], [u.y], marker="x", s=55)
                continue
            ax.scatter([u.x], [u.y], marker=marker[u.uav_type], s=65)
            ax.text(u.x + 20, u.y + 20, f"U{u.idx}:{u.uav_type}")

            if show_ranges:
                ax.add_patch(plt.Circle((u.x, u.y), u.p["recon_range"],
                                        fill=False, alpha=0.15))
                ax.add_patch(plt.Circle((u.x, u.y), u.p["comm_range"],
                                        fill=False, alpha=0.10, linestyle=":"))

        if show_comm:
            g = self.communication_graph()
            drawn = set()
            for a, ns in g.items():
                for b in ns:
                    e = tuple(sorted((a, b)))
                    if e in drawn:
                        continue
                    drawn.add(e)
                    ua, ub = self.uavs[a], self.uavs[b]
                    ax.plot([ua.x, ub.x], [ua.y, ub.y], linestyle="-.", linewidth=0.7)

        return ax
