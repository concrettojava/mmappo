import math
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge, Circle
from .style import TRAIL_LEN, TARGET_COLOR, THREAT_COLOR, TYPE_MARKER, UAV_COLORS


class SceneRenderer:
    """Draw a scene without advancing its simulation."""

    def __init__(self, env):
        self.env = env
        self.fig, self.ax = plt.subplots(figsize=(9.8, 7.8))
        self.event_text = ""
        self.reset_tracks()

    def reset_tracks(self):
        self.uav_tracks = {
            u.idx: deque([(u.x, u.y)], maxlen=TRAIL_LEN) for u in self.env.uavs
        }
        self.target_tracks = {
            t.idx: deque([(t.x, t.y)], maxlen=TRAIL_LEN) for t in self.env.targets
        }
        self.event_text = ""

    def set_event_text(self, text: str | None):
        self.event_text = text or ""

    def record_tracks(self):
        # Once an entity is destroyed/lost, keep its last real trajectory instead
        # of appending the same frozen point on every later animation frame.
        for u in self.env.uavs:
            if u.alive:
                self.uav_tracks[u.idx].append((u.x, u.y))
        for t in self.env.targets:
            if t.alive:
                self.target_tracks[t.idx].append((t.x, t.y))

    def _heading_pointer(
        self,
        x,
        y,
        yaw,
        color,
        start_offset=12.0,
        line_length=68.0,
        head_length=14.0,
        lw=1.0,
        z=12,
    ):
        ux = math.sin(yaw)
        uy = math.cos(yaw)

        x0 = x + start_offset * ux
        y0 = y + start_offset * uy

        x1 = x0 + line_length * ux
        y1 = y0 + line_length * uy

        self.ax.plot(
            [x0, x1],
            [y0, y1],
            color=color,
            linewidth=lw,
            alpha=0.95,
            solid_capstyle="round",
            zorder=z,
        )

        phi = math.radians(24.0)
        a1 = yaw + math.pi - phi
        a2 = yaw + math.pi + phi

        hx1 = x1 + head_length * math.sin(a1)
        hy1 = y1 + head_length * math.cos(a1)
        hx2 = x1 + head_length * math.sin(a2)
        hy2 = y1 + head_length * math.cos(a2)

        self.ax.plot(
            [hx1, x1, hx2],
            [hy1, y1, hy2],
            color=color,
            linewidth=lw,
            alpha=0.95,
            solid_capstyle="round",
            zorder=z,
        )

    def _draw_fading_track(self, points, base_rgb, max_alpha=0.34, lw=1.25):
        pts = np.asarray(points, dtype=float)
        if len(pts) < 2:
            return

        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        n = len(segs)
        alphas = np.linspace(0.035, max_alpha, n)

        colors = np.zeros((n, 4), dtype=float)
        colors[:, :3] = base_rgb
        colors[:, 3] = alphas

        self.ax.add_collection(
            LineCollection(segs, colors=colors, linewidths=lw, zorder=2)
        )

    def _component_sets(self):
        per_uav = self.env.communication_components()
        unique = []
        seen = set()

        for comp in per_uav.values():
            key = tuple(sorted(comp))
            if key not in seen:
                seen.add(key)
                unique.append(set(comp))
        return unique

    def _draw_union_contour(
        self,
        member_ids,
        radius_key,
        linestyle,
        alpha,
        linewidth,
    ):
        member_ids = [uid for uid in member_ids if self.env.uavs[uid].alive]
        if not member_ids:
            return

        xs = np.linspace(0, self.env.world_size, 175)
        ys = np.linspace(0, self.env.world_size, 175)
        X, Y = np.meshgrid(xs, ys)
        field = np.full_like(X, np.inf, dtype=float)

        for uid in member_ids:
            u = self.env.uavs[uid]
            r = u.p[radius_key]
            d = np.sqrt((X - u.x) ** 2 + (Y - u.y) ** 2) - r
            field = np.minimum(field, d)

        self.ax.contour(
            X,
            Y,
            field,
            levels=[0],
            colors=["#8a8a8a"],
            linewidths=[linewidth],
            linestyles=[linestyle],
            alpha=alpha,
            zorder=1,
        )

    def _draw_united_ranges(self):
        for comp in self._component_sets():
            self._draw_union_contour(comp, "recon_range", "--", 0.44, 0.9)
            self._draw_union_contour(comp, "comm_range", ":", 0.44, 0.9)

    def _draw_communication_topology(self):
        graph = self.env.communication_graph()
        done = set()

        for a, neighbors in graph.items():
            for b in neighbors:
                if not self.env.uavs[a].alive or not self.env.uavs[b].alive:
                    continue
                edge = tuple(sorted((a, b)))
                if edge in done:
                    continue
                done.add(edge)

                ua = self.env.uavs[a]
                ub = self.env.uavs[b]
                self.ax.plot(
                    [ua.x, ub.x],
                    [ua.y, ub.y],
                    color="#7a7a7a",
                    linestyle="-.",
                    linewidth=0.85,
                    alpha=0.45,
                    zorder=3,
                )

    def _draw_strike_sector(self, u):
        p = u.p
        center_deg = 90.0 - math.degrees(u.yaw)
        half = p["strike_angle_deg"] / 2.0
        color = UAV_COLORS[u.idx]

        self.ax.add_patch(
            Wedge(
                (u.x, u.y),
                p["strike_range"],
                center_deg - half,
                center_deg + half,
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.11,
                zorder=1,
            )
        )

    def _draw_destroyed_marker(self, x, y, label):
        # Persistent wreck/destruction marker.  A ring keeps it visible over
        # dense trajectories while the X makes the state unambiguous.
        self.ax.add_patch(
            Circle(
                (x, y),
                55.0,
                facecolor="none",
                edgecolor="#555555",
                linewidth=1.0,
                linestyle="--",
                alpha=0.55,
                zorder=14,
            )
        )
        self.ax.scatter(
            x,
            y,
            marker="x",
            s=90,
            color="#333333",
            linewidths=2.0,
            zorder=15,
        )
        self.ax.text(
            x + 24,
            y + 28,
            label,
            fontsize=7.6,
            color="#444444",
            weight="bold",
            zorder=16,
        )

    def _draw_threats(self):
        for th in self.env.threats:
            self.ax.add_patch(
                Circle(
                    (th.x, th.y),
                    th.radius,
                    facecolor=THREAT_COLOR,
                    edgecolor="none",
                    alpha=0.075,
                    zorder=0,
                )
            )
            self.ax.scatter(
                th.x,
                th.y,
                marker="X",
                s=62,
                color=THREAT_COLOR,
                alpha=0.92,
                zorder=11,
            )
            self.ax.text(
                th.x + 20,
                th.y + 22,
                f"T{th.idx}",
                fontsize=8,
                color=THREAT_COLOR,
                zorder=12,
            )

    def _draw_targets(self):
        for t in self.env.targets:
            self._draw_fading_track(
                self.target_tracks[t.idx],
                base_rgb=(0.18, 0.18, 0.18),
                max_alpha=0.32,
                lw=1.1,
            )

            if not t.alive:
                self._draw_destroyed_marker(t.x, t.y, f"M{t.idx} destroyed")
                continue

            self.ax.scatter(
                t.x,
                t.y,
                marker="s",
                s=30,
                color=TARGET_COLOR,
                zorder=10,
            )
            self._heading_pointer(
                t.x,
                t.y,
                t.yaw,
                color=TARGET_COLOR,
                start_offset=10,
                line_length=58,
                head_length=12,
                lw=0.9,
                z=12,
            )
            self.ax.text(
                t.x + 18,
                t.y + 22,
                f"M{t.idx}",
                fontsize=8,
                color="#222222",
                zorder=13,
            )

    def _draw_uavs(self):
        for u in self.env.uavs:
            color = UAV_COLORS[u.idx]
            rgb = tuple(
                int(color.lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4)
            )
            self._draw_fading_track(
                self.uav_tracks[u.idx],
                base_rgb=rgb,
                max_alpha=0.30,
                lw=1.25,
            )

            if not u.alive:
                self._draw_destroyed_marker(u.x, u.y, f"U{u.idx} lost")
                continue

            self._draw_strike_sector(u)
            self.ax.scatter(
                u.x,
                u.y,
                marker=TYPE_MARKER[u.uav_type],
                s=56,
                color=color,
                edgecolors="none",
                zorder=10,
            )
            self._heading_pointer(
                u.x,
                u.y,
                u.yaw,
                color=color,
                start_offset=12,
                line_length=68,
                head_length=14,
                lw=1.0,
                z=13,
            )
            self.ax.text(
                u.x + 16,
                u.y + 18,
                f"U{u.idx}",
                fontsize=7.4,
                color=color,
                weight="bold",
                zorder=14,
            )

    def _draw_legend(self):
        legend = [
            Line2D([0], [0], color="#8a8a8a", linestyle=":", lw=0.9,
                   label="United Communication Range"),
            Line2D([0], [0], color="#8a8a8a", linestyle="--", lw=0.9,
                   label="United Reconnaissance Range"),
            Line2D([0], [0], color="#999999", lw=5, alpha=0.25,
                   label="Strike Range"),
            Line2D([0], [0], color="#7a7a7a", linestyle="-.", lw=0.9,
                   label="Communication Topology"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="#777777",
                   markersize=7, label="Strike-Enhanced RSUAV"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#777777",
                   markersize=7, label="Communication-Enhanced RSUAV"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#777777",
                   markersize=6, label="Reconnaissance-Enhanced RSUAV"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=TARGET_COLOR,
                   markersize=6, label="Enemy Mobile Target"),
            Line2D([0], [0], marker="X", color="w", markerfacecolor=THREAT_COLOR,
                   markersize=8, label="Enemy Threatened Area"),
            Line2D([0], [0], marker="x", color="#333333", linestyle="None",
                   markersize=7, markeredgewidth=1.8, label="Destroyed / Lost Entity"),
        ]

        self.ax.legend(
            handles=legend,
            loc="upper left",
            bbox_to_anchor=(1.015, 1.0),
            frameon=False,
            fontsize=7.4,
            borderaxespad=0,
        )

    def _draw_hud(self):
        alive_uavs = sum(u.alive for u in self.env.uavs)
        alive_targets = sum(t.alive for t in self.env.targets)
        total_uavs = len(self.env.uavs)
        total_targets = len(self.env.targets)
        completion = 1.0 - alive_targets / max(total_targets, 1)
        survival = alive_uavs / max(total_uavs, 1)

        self.ax.text(
            0.015,
            0.985,
            f"t = {self.env.step_count:3d} s",
            transform=self.ax.transAxes,
            va="top",
            fontsize=8.5,
            color="#333333",
        )
        self.ax.text(
            0.985,
            0.985,
            f"UAVs {alive_uavs}/{total_uavs}   Targets {alive_targets}/{total_targets}\n"
            f"Completion {completion:.0%}   Survival {survival:.0%}",
            transform=self.ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.8,
            color="#333333",
        )
        if self.event_text:
            self.ax.text(
                0.5,
                0.945,
                self.event_text,
                transform=self.ax.transAxes,
                ha="center",
                va="top",
                fontsize=8.0,
                color="#333333",
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="#bbbbbb",
                    alpha=0.88,
                ),
                zorder=30,
            )

    def draw(self):
        self.ax.clear()
        self.ax.set_xlim(0, 4000)
        self.ax.set_ylim(0, 4000)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("x/m", fontsize=9)
        self.ax.set_ylabel("y/m", fontsize=9)
        self.ax.set_xticks(np.arange(0, 4001, 500))
        self.ax.set_yticks(np.arange(0, 4001, 500))
        self.ax.tick_params(labelsize=8)
        self.ax.grid(False)

        self._draw_united_ranges()
        self._draw_communication_topology()
        self._draw_threats()
        self._draw_targets()
        self._draw_uavs()
        self._draw_legend()
        self._draw_hud()

        self.ax.text(
            0.015,
            0.015,
            "SPACE pause/resume   R reset   Q/ESC quit",
            transform=self.ax.transAxes,
            va="bottom",
            fontsize=7.2,
            color="#666666",
        )

        self.fig.subplots_adjust(left=0.085, right=0.72, bottom=0.085, top=0.975)
