import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from .renderer import SceneRenderer
from .style import TARGET_COLOR, THREAT_COLOR, TYPE_MARKER, UAV_COLORS


JAMMER_COLOR = "#7b2cbf"
COMM_COLOR = "#1696d2"
RECON_COLOR = "#2ca25f"
UAV_LOST_COLOR = "#d62728"
TARGET_DESTROYED_COLOR = "#222222"


class ContestedSceneRenderer(SceneRenderer):
    """Renderer for the contested scenario.

    The environment mechanics stay authoritative; this class only exposes the
    new contested-scenario state visually: jammer fields, local interference,
    and the difference between nominal and currently effective sensing/
    communication ranges.
    """

    def _draw_jammers(self):
        for idx, jammer in enumerate(getattr(self.env, "jammers", [])):
            # The environment uses jammer['radius'] as the Gaussian spatial
            # scale.  Draw that scale as the visible influence footprint while
            # opacity follows the jammer's current temporal activity.
            temporal = 0.65 + 0.35 * np.sin(
                2.0 * np.pi * self.env.step_count / jammer["period"] + jammer["phase"]
            )
            activity = float(np.clip(jammer["strength"] * temporal, 0.0, 1.0))
            alpha = 0.05 + 0.12 * activity
            self.ax.add_patch(
                Circle(
                    (jammer["x"], jammer["y"]),
                    jammer["radius"],
                    facecolor=JAMMER_COLOR,
                    edgecolor=JAMMER_COLOR,
                    linewidth=1.1,
                    linestyle="--",
                    alpha=alpha,
                    zorder=0,
                )
            )
            self.ax.scatter(
                jammer["x"],
                jammer["y"],
                marker="P",
                s=72,
                color=JAMMER_COLOR,
                alpha=0.92,
                zorder=8,
            )
            self.ax.text(
                jammer["x"] + 22,
                jammer["y"] + 26,
                f"J{idx}",
                fontsize=8,
                color=JAMMER_COLOR,
                weight="bold",
                zorder=9,
            )

    def _draw_effective_ranges(self):
        for u in self.env.uavs:
            if not u.alive:
                continue

            comm_nominal = float(u.p["comm_range"])
            recon_nominal = float(u.p["recon_range"])
            comm_effective = comm_nominal * self.env.communication_factor(u)
            recon_effective = recon_nominal * self.env.reconnaissance_factor(u)

            # Faint nominal circles make the compression directly visible.
            self.ax.add_patch(
                Circle(
                    (u.x, u.y),
                    comm_nominal,
                    facecolor="none",
                    edgecolor="#9a9a9a",
                    linewidth=0.55,
                    linestyle=":",
                    alpha=0.18,
                    zorder=0,
                )
            )
            self.ax.add_patch(
                Circle(
                    (u.x, u.y),
                    recon_nominal,
                    facecolor="none",
                    edgecolor="#9a9a9a",
                    linewidth=0.55,
                    linestyle="--",
                    alpha=0.18,
                    zorder=0,
                )
            )

            # Current effective ranges are the physically relevant radii used
            # by communication/detection in the contested environment.
            self.ax.add_patch(
                Circle(
                    (u.x, u.y),
                    comm_effective,
                    facecolor="none",
                    edgecolor=COMM_COLOR,
                    linewidth=0.75,
                    linestyle=":",
                    alpha=0.32,
                    zorder=1,
                )
            )
            self.ax.add_patch(
                Circle(
                    (u.x, u.y),
                    recon_effective,
                    facecolor="none",
                    edgecolor=RECON_COLOR,
                    linewidth=0.75,
                    linestyle="--",
                    alpha=0.32,
                    zorder=1,
                )
            )

    @staticmethod
    def _interference_color(intensity: float) -> str:
        if intensity < 0.25:
            return "#f6c945"
        if intensity < 0.50:
            return "#f28e2b"
        return "#d62728"

    def _draw_target_destroyed(self, target):
        self.ax.add_patch(
            Circle(
                (target.x, target.y),
                55.0,
                facecolor="none",
                edgecolor=TARGET_DESTROYED_COLOR,
                linewidth=0.9,
                linestyle="--",
                alpha=0.42,
                zorder=14,
            )
        )
        self.ax.scatter(
            target.x,
            target.y,
            marker="x",
            s=90,
            color=TARGET_DESTROYED_COLOR,
            linewidths=2.0,
            zorder=15,
        )
        self.ax.text(
            target.x + 24,
            target.y + 28,
            f"M{target.idx} destroyed",
            fontsize=7.6,
            color=TARGET_DESTROYED_COLOR,
            weight="bold",
            zorder=16,
        )

    def _draw_uav_lost(self, uav):
        self.ax.add_patch(
            Circle(
                (uav.x, uav.y),
                58.0,
                facecolor="none",
                edgecolor=UAV_LOST_COLOR,
                linewidth=1.0,
                linestyle="--",
                alpha=0.52,
                zorder=14,
            )
        )
        self.ax.scatter(
            uav.x,
            uav.y,
            marker="x",
            s=105,
            color=UAV_LOST_COLOR,
            linewidths=2.4,
            zorder=15,
        )
        self.ax.text(
            uav.x + 24,
            uav.y + 28,
            f"U{uav.idx} lost",
            fontsize=7.6,
            color=UAV_LOST_COLOR,
            weight="bold",
            zorder=16,
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
                self._draw_target_destroyed(t)
                continue

            self.ax.scatter(t.x, t.y, marker="s", s=30, color=TARGET_COLOR, zorder=10)
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
                self._draw_uav_lost(u)
                continue

            intensity = self.env.jammer_intensity_at(u.x, u.y)
            if intensity > 0.02:
                self.ax.add_patch(
                    Circle(
                        (u.x, u.y),
                        72.0,
                        facecolor="none",
                        edgecolor=self._interference_color(intensity),
                        linewidth=1.8,
                        alpha=0.45 + 0.45 * intensity,
                        zorder=9,
                    )
                )

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
                f"U{u.idx}  I={intensity:.2f}",
                fontsize=7.2,
                color=color,
                weight="bold",
                zorder=14,
            )

    def _draw_legend(self):
        legend = [
            Line2D([0], [0], marker="P", color="w", markerfacecolor=JAMMER_COLOR,
                   markersize=8, label="Jammer"),
            Line2D([0], [0], color="#9a9a9a", linestyle=":", lw=0.8,
                   label="Nominal Range"),
            Line2D([0], [0], color=COMM_COLOR, linestyle=":", lw=1.0,
                   label="Effective Communication Range"),
            Line2D([0], [0], color=RECON_COLOR, linestyle="--", lw=1.0,
                   label="Effective Reconnaissance Range"),
            Line2D([0], [0], color="#7a7a7a", linestyle="-.", lw=0.9,
                   label="Communication Topology"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="#777777",
                   markersize=7, label="Strike-Enhanced RSUAV"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#777777",
                   markersize=6, label="Reconnaissance-Enhanced RSUAV"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#777777",
                   markersize=7, label="Communication-Enhanced RSUAV"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=TARGET_COLOR,
                   markersize=6, label="Enemy Mobile Target"),
            Line2D([0], [0], marker="X", color="w", markerfacecolor=THREAT_COLOR,
                   markersize=8, label="Enemy Threatened Area"),
            Line2D([0], [0], marker="x", color=TARGET_DESTROYED_COLOR,
                   linestyle="None", markersize=7, markeredgewidth=1.8,
                   label="Destroyed Target"),
            Line2D([0], [0], marker="x", color=UAV_LOST_COLOR,
                   linestyle="None", markersize=7, markeredgewidth=2.0,
                   label="Lost UAV"),
        ]
        self.ax.legend(
            handles=legend,
            loc="upper left",
            bbox_to_anchor=(1.015, 1.0),
            frameon=False,
            fontsize=7.2,
            borderaxespad=0,
        )

    def _draw_hud(self):
        super()._draw_hud()
        self.ax.text(
            0.015,
            0.945,
            "scenario = contested",
            transform=self.ax.transAxes,
            va="top",
            fontsize=7.4,
            color=JAMMER_COLOR,
            weight="bold",
        )

    def draw(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.env.world_size)
        self.ax.set_ylim(0, self.env.world_size)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("x/m", fontsize=9)
        self.ax.set_ylabel("y/m", fontsize=9)
        self.ax.set_xticks(np.arange(0, self.env.world_size + 1, 500))
        self.ax.set_yticks(np.arange(0, self.env.world_size + 1, 500))
        self.ax.tick_params(labelsize=8)
        self.ax.grid(False)

        self._draw_jammers()
        self._draw_effective_ranges()
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
