# FOFE-MMAPPO 场景工程

已完成代码模块化，尚未实现奖励和强化学习算法，论文规则待逐项核查。

## 目录职责

- `src/fofe_mmapppo/envs/`：entities（实体与参数）、scenario（场景初始化）、dynamics（运动）、communication（通信与共享探测）、combat（打击与损失）、observation（初版观测和状态）、uav_env（环境组装）。
- `src/fofe_mmapppo/visualization/`：style（样式）、renderer（绘图和轨迹）、live_viewer（动态预览和交互）。
- `scripts/view_scene.py`：统一查看和截图入口。
- `tests/check_regression.py`：与原始基准对照。
- `legacy/baseline_20260916/`：原始代码与图片备份，不再修改。
- `outputs/baseline_20260916/`：冻结的轨迹、图片及校验值。
- `outputs/checks/`：检查输出，可删除。
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

## 检查

```bash
.venv/bin/python tests/check_regression.py
```

对照 6 组完整 200 步环境输出、50 步预览状态、两张逐像素一致的图片，并检查键盘回调。基准版本为 Python 3.12.3、NumPy 2.5.3、Matplotlib 3.11.2，图像比较依赖相同绘图库和字体环境。回归通过表示行为保持一致，不证明论文规则正确。

初始化绘图现在不会额外推进动画。旧根目录 Python 入口和简易 `env.render()` 已移除，绘图统一由 `SceneRenderer` 负责。

## 开发

现有入口可直接运行。如需安装项目包，在目标 Python 环境执行 `python -m pip install -e .`。以后只修改 `src/` 中的正式实现。下一步核查论文，完善 observation/state、reward 和规则测试。
