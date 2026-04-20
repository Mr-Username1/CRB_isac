# 项目整体说明

## 1. 项目目标

本项目用于复现论文 *ISAC from the Sky: UAV Trajectory Design for Joint Communication and Target Localization* 的核心流程，重点包括：

- UAV 多阶段轨迹优化（通信-感知权衡）；
- 感知测距观测建模与 MLE 目标定位；
- 能耗约束下的阶段推进与末阶段自适应；
- 结果保存与离线画图分析。

## 2. 当前结构（重构后）

- `system_model.py`
  - 系统模型、通信/感知链路、CRB 指标与场景参数预设。
- `problem.py`
  - UAV 推进能耗模型与阶段能耗计算。
- `p2_solver.py`
  - 单阶段 `P2(m)` 求解器（SCA + 凸子问题）。
- `simulation_pipeline.py`
  - 仿真主流程库（粗扫描、量测仿真、MLE、多阶段循环、方法运行）。
- `config_factory.py`
  - 统一构建 `cfg/e_cfg/scfg`，避免脚本里散落全局配置。
- `io_utils.py`
  - 结果保存工具（`npz` + `meta json`）。
- `generate_results.py`
  - 推荐的唯一实验入口（批量运行并保存结果）。
- `plot_saved_results.py`
  - 离线绘图入口（读取 `results/isac_results.npz`）。
- `main.py`
  - 轻量单案例演示入口（便于快速 smoke test）。

## 3. 流程说明

### 3.1 初始粗扫描

1. 生成粗扫描悬停点；
2. 按场景参数模拟测距观测（支持高噪声、NLOS 偏置、异常点）；
3. 两级网格 MLE 获取初始目标估计；
4. 扣减粗扫描能耗并作为多阶段初始状态。

### 3.2 多阶段优化与估计

每个阶段：

1. 调用 `solve_p2m_sca(...)` 进行轨迹优化；
2. 计算阶段能耗并更新剩余能量；
3. 累计测距样本并做 MLE 更新；
4. 记录阶段日志（CRB、Rate、求解器状态、估计更新）。

`P2(m)` 的通信项按累计口径实现（Eq.33 思路）：
- 使用历史 `R_prev_sum/N_prev_total` 与当前阶段速率共同构成目标。

## 4. 使用方法

1. 生成结果（推荐入口）：

```bash
python generate_results.py
```

2. 绘图并保存：

```bash
python plot_saved_results.py
```

3. 单案例快速验证：

```bash
python main.py
```

图像默认保存到 `results/figures/`。

## 5. 当前边界

- 目标估计更新为“阶段级更新”（每阶段一次）。
- 若要严格对齐论文完整统计图（Fig.5/6/8/9 风格），仍需补充 Monte Carlo 批量扫参与统计脚本。
