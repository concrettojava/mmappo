import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ..envs import CooperativeUAVEnv
from .renderer import SceneRenderer

class LiveViewer:
    """The original v5 movement-only preview, with no combat or episode end."""
    def __init__(self, seed=7, interval_ms=120):
        self.env = CooperativeUAVEnv(seed=seed)
        self.env.reset()
        self.renderer = SceneRenderer(self.env)
        self.fig = self.renderer.fig
        self.paused = False
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.anim = FuncAnimation(self.fig, self._animate, interval=interval_ms,
                                  blit=False, cache_frame_data=False,
                                  init_func=self.renderer.draw)

    def _on_key(self, e):
        if e.key == " ":
            self.paused = not self.paused
        elif e.key in ("r", "R"):
            self.env.reset()
            self.renderer.reset_tracks()
            self.paused = False
        elif e.key in ("q", "Q", "escape"):
            plt.close(self.fig)

    def _random_uav_actions(self):
        actions = []
        for u in self.env.uavs:
            r = self.env.rng.random()

            if r < 0.025:
                a = 1
            elif r < 0.10:
                a = 2
            elif r > 0.975:
                a = 5
            elif r > 0.90:
                a = 4
            else:
                a = 3

            margin = 280

            if u.x < margin:
                a = 4
            elif u.x > self.env.world_size - margin:
                a = 2

            if u.y < margin and math.cos(u.yaw) < 0:
                a = 4 if self.env.rng.random() < 0.5 else 2
            elif u.y > self.env.world_size - margin and math.cos(u.yaw) > 0:
                a = 4 if self.env.rng.random() < 0.5 else 2

            actions.append(a)

        return np.asarray(actions, dtype=int)

    def _advance_scene_only(self):
        self.env.step_count += 1
        self.env._update_uavs(self._random_uav_actions())
        self.env._update_targets()

        for u in self.env.uavs:
            if u.x < 0 or u.x > self.env.world_size:
                u.yaw = -u.yaw
                u.x = float(np.clip(u.x, 0, self.env.world_size))

            if u.y < 0 or u.y > self.env.world_size:
                u.yaw = math.pi - u.yaw
                u.y = float(np.clip(u.y, 0, self.env.world_size))

            u.yaw = (u.yaw + math.pi) % (2 * math.pi) - math.pi

        self.renderer.record_tracks()

    def _animate(self, _):
        if not self.paused:
            self._advance_scene_only()

        self.renderer.draw()
        return []

    def show(self):
        self.renderer.draw()
        plt.show()
