# PI-MAPPO independent actor batching (2026-09-18)

## Scope

Worktree: `/home/fofe_mmapppo_pi`, branch `codex/pi-batched-update`.
Base revision: `f97b47e`.
GPU: RTX 3080, 10 GiB, WSL Ubuntu. PyTorch `2.14.0+cu130`.

The CUDA training CLI now defaults to batched actor updates; CPU retains sequential
updates. `--no-batch-actor-updates` selects the original update path.
`--actor-batch-size` controls the number of independent actors executed together
(default 8). The library config remains opt-in with `batch_actor_updates=True`.

Each actor still has its own weights, Adam optimizer state, advantage normalization,
environment permutation, and gradient norm clipping. Critics remain independent
centralized MLPs. The default TBPTT span remains 50 steps, with behavior-policy
boundary states reused across PPO epochs as in the original trainer. Checkpoint
keys and optimizer states remain compatible with the sequential implementation.

`pi_batched.py` uses differentiable parameter stacking and `torch.func.vmap`.
One recurrent step is compiled with `fullgraph=True, dynamic=False`; Python
retains the time loop. Static variants handle rollout/replay batch-size changes.
Automatic dynamic-shape compilation was observed to fall back to eager execution
in real training; an experimental full-graph dynamic compile also crashed inside
symbolic-shape processing. The validated path uses static shapes. Non-reentrant
activation checkpointing recomputes intermediate activations during backward,
without detaching the recurrent state within a chunk. All temporal chunks contribute
to one optimizer step per environment minibatch. No optimizer step is taken for
an actor without active samples in that minibatch.

## Findings

The original whole-sequence compile was interrupted after more than five minutes
without timing output. Step-level compilation without activation checkpointing
then ran out of memory in the eight-actor backward warmup. The error reported
24.12 GiB of PyTorch allocations under WSL; this exceeds physical GPU capacity
and is not a viable configuration on this card.

With step-level compilation and activation checkpointing, the requested
8 actors x 4 environments x 50 timesteps benchmark completed (2 warmups, 5 measured
iterations):

| Measurement | Sequential | Batched + recomputation |
| --- | ---: | ---: |
| Forward + backward | 6.4202 s | 1.6576 s |
| PyTorch peak allocated | 1483.6 MiB | 682.5 MiB |
| Loss | 0.01629462 | 0.01629462 |

Speedup: **3.873x**. Maximum logits difference: `1.192093e-07`.
Maximum recurrent-state difference: `1.192093e-05`.
The sequential baseline now backpropagates one actor at a time, matching the
trainer's memory lifetime. Both paths use the same loss scaling. These are
PyTorch allocation peaks, not total device usage reported by nvidia-smi.
Raw log: `outputs/codex_optimization/sequence_checkpoint.log`.

Floating-point reduction order changes under batching, so bitwise identity is
not expected. Adam can magnify roundoff in almost-zero gradients, especially
softmax-invariant biases. Tests therefore cover float32 logits, state, gradients
and post-update policy outputs, plus strict float64 multi-epoch parameter and
optimizer-state comparisons. This is numerical/engineering validation, not a
claim of identical long-run learning curves.

## Reproduce

```bash
cd /home/fofe_mmapppo_pi
PY=/home/fofe_mmapppo_scene_stage1/.venv/bin/python
$PY scripts/bench_pi_multi_actor_sequence.py \
  --device cuda --agents 8 --batch 4 --steps 50 \
  --compile-sequential --compile-vmap --warmup 2 --iters 5

$PY scripts/train_pi_mappo_parallel.py \
  --device cuda --episodes 16 --num-envs 8 --max-steps 200 \
  --ppo-epochs 3 --sequence-env-minibatch-size 4 \
  --replay-check-every 1 --log-every 8 --save-every 16 \
  --output outputs/pi_batched_validation

PYTHONPATH=src OMP_NUM_THREADS=1 $PY -m unittest discover -s tests -v
```

`PYTHONPATH=src` is necessary when using the other worktree's virtual environment
for test discovery. The training and benchmark entry points insert this worktree's
source path themselves. No virtual-environment installation was changed.

For diagnostic comparisons only, the benchmark retains
`--vmap-compile-scope sequence --no-activation-checkpoint` and
`--no-activation-checkpoint`. Those modes can require much more time or memory.

## Validation

- Full regression suite: **76 tests passed**.
- New tests: full BPTT, chunked BPTT, partial final chunk, uneven environment
  pools, padded minibatches, absent actors, separate actor groups, float32
  gradients/policy outputs, and checkpoint/optimizer resume equivalence.
- Tiny CPU benchmark with the gradient-reporting code: max gradient error
  `1.862645e-09`, relative gradient L2 error `1.227661e-07`.
- Full regression log: `outputs/codex_optimization/final_all_tests.log`.

## Real CUDA training

Completed 16 episodes using 8 environments, 200 steps, 3 PPO epochs,
4 environments per minibatch, 50-step TBPTT and 8 actors per update group.
The normal CUDA defaults select the batched backend.

| Steady second batch | Previous local sequential run | Optimized run |
| --- | ---: | ---: |
| PPO update | 185.7183 s | 34.1783 s |
| GPU average utilization during update | 33.9% | 68.8% |
| PyTorch peak allocation during update | 1532.7 MiB | 754.9 MiB |
| Rollout collection | 12.2047 s | 14.4058 s |
| Explicit replay correctness check | Disabled | 9.2354 s |

Update speedup: **5.434x**. The optimized first batch took 77.6 seconds to
update, including additional compilation. The two optimized replay checks both
reported zero maximum ratio error. Rewards and displayed losses in these two
batches agree with the previous local run; this does not establish long-run
bitwise or convergence equivalence.

This comparison uses the existing baseline in
`outputs/pi_optimized_ppo3_env8_200/metrics.jsonl`, on the same GPU and training
configuration, rather than a newly randomized A/B timing experiment. New metrics:
`outputs/codex_optimization/train_static/metrics.jsonl`. The optimized run includes
extra replay checks, so its 5.434x update speedup should not be described as an
end-to-end speedup. End-to-end collection + replay + update is 57.82 seconds per
8-episode batch, versus 197.92 seconds in the previous run (about 3.42x, even
with the extra check). Cold compilation is excluded from that steady-state ratio.

The training validation wrapper only prints phase/unroll timing and periodic
stack diagnostics; model, optimizer and data flow are the normal training CLI.
A separate direct CLI run verifies resuming the user's original checkpoint.

Legacy checkpoint resume completed through the direct CLI: original episode 16
checkpoint -> episode 24. Actor and critic optimizer step counters reached 18;
all saved weights and logged metrics are finite, and replay maximum ratio error
is zero. Output: `outputs/codex_optimization/resume_legacy/checkpoint_000024.pt`.
Original checkpoints and experiment outputs were retained.

## Final static-backend gradient check

A final GPU run with 8 actors, 4 environments and 50 steps (1 warmup, 3 measured
iterations) measured 6.1495 s sequential versus 1.5399 s batched, a 3.994x speedup.
Maximum gradient absolute difference was `6.350456e-08`; gradient relative L2
error was `1.103661e-06`. Losses matched to the eight decimal places printed.
Peak PyTorch allocation was 1483.6 MiB sequential versus 688.6 MiB batched.
The latter includes copies of reference gradients retained for comparison.
Raw log: `outputs/codex_optimization/final_sequence_b4.log`.

## Using additional GPU memory

A follow-up sequence benchmark increased the environment minibatch from 4 to 8
while retaining 8 independent actors, 50 steps, activation checkpointing and
the same network. Both final runs used 1 warmup and 3 measured iterations.

| Batched sequence | 4 environments | 8 environments |
| --- | ---: | ---: |
| Forward + backward | 1.5399 s | 2.2482 s |
| Actor transitions per second | 1039 | 1423 |
| PyTorch peak allocated | 688.6 MiB | 1317.9 MiB |

Doubling minibatch size improved normalized sample throughput by **37.0%**,
while using about 91% more activation/measurement memory. It did not require
filling the 10 GiB card. The new batch-size variant had a substantial one-time
compilation cost; the table reports warmed execution only.

For the 8-environment benchmark, maximum logits difference was `2.086163e-07`,
maximum recurrent-state difference `2.391338e-04`, maximum gradient difference
`1.797453e-07`, and relative gradient L2 error `6.908368e-06`. Both paths printed
loss `0.01512962`. Raw log: `outputs/codex_optimization/final_sequence_b8.log`.

The production PPO environment minibatch default remains 4. Increasing it to 8
changes the number of optimizer steps and the gradient estimator, so this is an
optional experiment (`--sequence-env-minibatch-size 8`), not part of the
semantics-preserving 5.434x update-speed claim above. VRAM occupancy itself is
not an optimization target; measure samples/second and learning outcomes.
