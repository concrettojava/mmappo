# FOFE-MMAPPO 场景工程

已完成代码模块化，并按论文逐项实现 Dec-POMDP 环境。当前已完成 Eq. (8)/(9) observation/state 与 Eq. (11)–(17) reward；强化学习算法尚未实现。

## 目录职责

- `src/fofe_mmapppo/envs/`
  - `entities.py`：实体与异构参数
  - `scenario.py`：场景初始化
  - `dynamics.py`：UAV/目标运动学
  - `communication.py`：通信拓扑、多跳子群、共享探测与威胁记忆
  - `combat.py`：自动打击、碰撞与威胁损失
  - `coordinates.py`：geo/body 坐标转换
  - `observation.py`：论文 Eq. (8) 风格的本地 flexible observation
  - `state.py`：论文 Eq. (9) 风格的 centralized global state
  - `reward.py`：论文 Eq. (11)–(17) reward 与 Section 5.1 系数
  - `uav_env.py`：环境组装
- `src/fofe_mmapppo/visualization/`：style、renderer、live_viewer。
- `scripts/view_scene.py`：统一查看和截图入口。
- `tests/check_regression.py`：冻结 stage-1 的物理/可视化回归。
- `tests/test_observation_state.py`：Eq. (8)/(9)、body coordinate、通信共享与威胁记忆的规则测试。
- `tests/test_reward.py`：Eq. (11)–(17) 与异构 reward 系数的规则测试。
- `legacy/baseline_20260916/`：原始代码与图片备份，不再修改。
- `outputs/baseline_20260916/`：冻结的 stage-1 轨迹、图片及校验值。
- `pyproject.toml`：统一管理包信息和依赖。

## 运行（WSL Ubuntu 终端）

```bash
cd /home/fofe_mmapppo_scene_stage1
.venv/bin/python scripts/view_scene.py
.venv/bin/python scripts/view_scene.py --seed 42
.venv/bin/python scripts/view_scene.py --seed 7 --steps 50 --save outputs/figures/scene.png
```

空格暂停/继续，R 生成新场景，Q 或 Esc 退出。R 继续使用随机数序列；重新启动并指定同一 seed 可复现初始场景。保存图片无需图形窗口。

此入口延续 v5 的仅运动预览，不进行战斗结算或 episode 终止，并保留展示用边界反弹。真实环境使用 `CooperativeUAVEnv.step()`。

## Eq. (8) 本地观测

每个存活 UAV 的本地观测保留四个可变长度通道：

```text
self / neighbors / targets / threats
```

其中 `neighbors` 来自当前多跳通信子群，`targets` 为子群任意成员直接侦察到的存活移动目标，`threats` 为子群当前侦察或历史记忆到的威胁区域。含位置/航向的信息同时提供 `geo` 和观察者自身 `body` 坐标表示。

当前工程沿用已验证的场景约定：`yaw=0` 指向北。body 坐标定义为 `x=前向`、`y=右向`，同时提供 `range` 和 `bearing` 作为 Eq. (4) 所需的极坐标派生量。

## Eq. (9) centralized state

`get_global_state()` 返回以每架 UAV 为观察参考系的完整状态：

```text
state[agent_id] -> uavs / neighbors / targets / threats
```

与 local observation 使用同一记录格式，但取消可观测性约束：包含全部 UAV、全部移动目标与全部威胁区域，供后续 centralized critic / FOFE 使用。

## Eq. (11)–(17) reward

`CooperativeUAVEnv.step()` 现在返回：

```text
observation, global_state, rewards, done, info
```

其中 `rewards` 是 `{agent_id: reward}`，`info["reward_breakdown"]` 记录 Mission / Ability / Action / Boundary 以及 Ability 内部的 Avoid / Destroy / Strike / Reconnaissance / Communication 分量。

Section 5.1 的论文系数已经固化在 `reward.py`：`lambda_time=1/100`、`lambda_dist=1/2000`、`lambda_Mission=1`、`lambda_Ability=1`、`lambda_Action=0.25`、`lambda_Bound=1`，以及 Stk/Rec/Com 三类 UAV 对三种能力 reward 的 0.7/0.2/0.1 等异构比例。

有两个实现时序约定被显式记录在 `reward.py`：

1. Ability reward 在运动与探测更新后、自动打击/碰撞/威胁损毁前采样几何状态。否则 Eq. (14) 的 `|A_i^M|` 会因“可打击目标立即被自动击毁”而恒为 0。
2. `R_Destroy=-50` 只在 UAV 本步新损毁时施加一次；已经死亡并退出决策过程的 UAV 后续 reward 为 0。

## 检查

Stage-1 物理与可视化回归：

```bash
.venv/bin/python tests/check_regression.py
```

Eq. (8)/(9) 规则测试：

```bash
.venv/bin/python -m unittest tests/test_observation_state.py -v
```

Eq. (11)–(17) reward 规则测试：

```bash
.venv/bin/python -m unittest tests/test_reward.py -v
```

`check_regression.py` 不比较新的 observation/state/reward API，而是比较冻结的底层实体世界状态、episode 信息、随机预览轨迹以及两张基准图片，确保新增 Dec-POMDP 表示和 reward 没有改变 stage-1 的运动/战斗/可视化行为。

## 开发

如需安装项目包：

```bash
python -m pip install -e .
```

下一步：在 reward 规则测试与 stage-1 regression 均通过后，实现固定长度 observation 的纯 MAPPO baseline；baseline 可训练后再接 FOFE 与 Mamba。
