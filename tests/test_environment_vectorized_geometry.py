import math
import unittest

from fofe_mmapppo.envs import CooperativeUAVEnv


class TestVectorizedCommunicationGeometry(unittest.TestCase):
    def setUp(self):
        self.env = CooperativeUAVEnv(seed=19)
        self.env.reset(seed=19)

    def slow_graph(self):
        alive = [u for u in self.env.uavs if u.alive]
        graph = {u.idx: set() for u in alive}
        for i, a in enumerate(alive):
            for b in alive[i + 1:]:
                d = math.hypot(a.x - b.x, a.y - b.y)
                if d < max(a.p["comm_range"], b.p["comm_range"]):
                    graph[a.idx].add(b.idx)
                    graph[b.idx].add(a.idx)
        return graph

    def slow_targets(self, uav):
        return {
            t.idx
            for t in self.env.targets
            if t.alive
            and math.hypot(uav.x - t.x, uav.y - t.y) < uav.p["recon_range"]
        }

    def slow_threats(self, uav):
        return {
            th.idx
            for th in self.env.threats
            if math.hypot(uav.x - th.x, uav.y - th.y) < uav.p["recon_range"]
        }

    def test_graph_matches_slow_reference(self):
        self.assertEqual(self.env.communication_graph(), self.slow_graph())

    def test_direct_detection_matches_slow_reference(self):
        for uav in self.env.uavs:
            if not uav.alive:
                continue
            self.assertEqual(self.env._direct_detected_targets(uav), self.slow_targets(uav))
            before = set(uav.threat_memory)
            expected = self.slow_threats(uav)
            actual = self.env._direct_detected_threats(uav)
            self.assertEqual(actual, expected)
            self.assertEqual(uav.threat_memory, before | expected)

    def test_reused_shared_detection_matches_fresh_path(self):
        graph = self.env.communication_graph()
        direct_targets = {
            u.idx: self.env._direct_detected_targets(u)
            for u in self.env.uavs if u.alive
        }
        direct_threats = {
            u.idx: self.env._direct_detected_threats(u)
            for u in self.env.uavs if u.alive
        }
        reused = self.env.shared_detection(
            graph=graph,
            direct_targets=direct_targets,
            direct_threats=direct_threats,
        )
        fresh = self.env.shared_detection()
        self.assertEqual(reused, fresh)


if __name__ == "__main__":
    unittest.main()
