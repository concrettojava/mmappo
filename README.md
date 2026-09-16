# FOFE-MMAPPO 场景工程

已完成代码模块化，并开始按论文逐项实现 Dec-POMDP 环境。奖励函数和强化学习算法尚未实现。

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
  - `uav_env.py`：环境组装
- `src/fofe_mmapppo/visualization/`：style、renderer、live_viewer。
- `scripts/view_scene.py`：统一查看和截图入口。
- `tests/check_regression.py`：冻结 stage-1 的物理/可视化回归。
- `tests/test_observation_state.py`：Eq. (8)/(9)、body coordinate、通信共享与威胁记忆的规则测试。
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

## 检查

Stage-1 物理与可视化回归：

```bash
.venv/bin/python tests/check_regression.py
```

Eq. (8)/(9) 规则测试：

```bash
.venv/bin/python -m unittest tests/test_observation_state.py -v
```

`check_regression.py` 不再比较新的 observation/state API，而是比较冻结的底层实体世界状态、episode 信息、随机预览轨迹以及两张基准图片，确保新增 Dec-POMDP 表示没有改变 stage-1 的运动/战斗逻辑。

## 开发

如需安装项目包：

```bash
python -m pip install -e .
```

下一步：按论文 Eq. (11)–(17) 实现完整 reward，并为每个奖励分量增加独立规则测试；之后才进入固定长度 MAPPO baseline 与 FOFE/MMAPPO。
