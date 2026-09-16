from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PITrainingScriptSmokeTests(unittest.TestCase):
    """Run the actual CLI trainer through collect -> PPO update -> checkpoint."""

    def test_tiny_cpu_training_run_writes_finite_metrics_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "train_pi_mappo_parallel.py"),
                "--episodes",
                "2",
                "--num-envs",
                "1",
                "--max-steps",
                "4",
                "--ppo-epochs",
                "1",
                "--sequence-env-minibatch-size",
                "1",
                "--horizon",
                "2",
                "--belief-dim",
                "8",
                "--context-dim",
                "8",
                "--relation-dim",
                "8",
                "--critic-hidden-dim",
                "16",
                "--replay-check-every",
                "1",
                "--log-every",
                "1",
                "--save-every",
                "2",
                "--device",
                "cpu",
                "--scenario",
                "contested",
                "--no-tensorboard",
                "--output",
                str(output),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stdout)

            metrics_path = output / "metrics.jsonl"
            checkpoint = output / "checkpoint_000002.pt"
            self.assertTrue(metrics_path.exists(), msg=completed.stdout)
            self.assertTrue(checkpoint.exists(), msg=completed.stdout)

            lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(lines), 2)
            records = [json.loads(line) for line in lines]
            self.assertEqual([record["episode"] for record in records], [1, 2])
            self.assertTrue(all(record["scenario"] == "contested" for record in records))
            self.assertTrue(all(record["replay_max_abs_ratio_error"] < 1e-4 for record in records))
            self.assertTrue(all(0.0 <= record["clip_fraction"] <= 1.0 for record in records))
            self.assertTrue(all(record["rollout_steps"] == 4 for record in records))


if __name__ == "__main__":
    unittest.main()
