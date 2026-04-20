# 修改日志

日期：2026-04-16

## 第一阶段：单阶段 P2(m) 对齐与稳定化

- `p2_solver.py`
  - 约束建模修正：`delta^-2` 改为 `cp.square(cp.inv_pos(delta[i]))`，提升 DCP 兼容性。
  - 新增一致性校验：`StageData.Km == floor(Nm/mu)`。
  - 新增阻尼更新机制（步长衰减到 `min_step_size`）以增强 SCA 数值稳定性。
  - 按要求改为让 `cvxpy` 自动选择求解器，不再强制 `SCS`。
  - 新增每迭代日志：打印 `solver/status/obj`。
  - 新增最终输出项：`solver_final`。

- `main.py`
  - 单阶段基线参数改为论文常用口径：`mu=5, Nm=25, Etot=35kJ`。
  - 增加最终输出打印：`status_final`、`solver_final`。

## 第二阶段：多阶段主循环骨架（未含 MLE）

- `problem.py`
  - 新增 `stage_energy_used(...)`：
    - 基于真实阶段起点计算单阶段飞行+悬停能耗。
    - 返回 `(E_used, K)` 供多阶段能量账本使用。

- `main.py`
  - 新增 `run_multistage_skeleton(...)`：
    - 实现 `m=1..M` 的分阶段优化循环。
    - 累计 `prev_hover_xy` 与阶段轨迹。
    - 更新下一阶段起点为上一阶段末端 waypoint。
    - 维护 `energy_left`，并输出阶段日志。

## 第三阶段：接入 MLE 与末阶段自适应

- `main.py`
  - 新增 `simulate_range_measurements(...)`：
    - 按论文测距模型和噪声方差公式，生成每阶段的测距观测。
  - 新增 `mle_grid_search(...)`：
    - 先粗网格后细网格，两级搜索最大似然点。
  - 将多阶段入口升级为 `run_multistage_with_mle(...)`：
    - 每阶段完成后累计测距样本。
    - 基于累计悬停点与累计测距执行 MLE，更新 `target_hat_xy`。
    - 增加末阶段自适应 `Nlst/Klst`：当 `Nstg` 不可行时，按 `mu` 递减尝试更短阶段。
    - 在阶段日志中记录 `target_hat_xy`。

## 当前状态说明

- 已完成：
  - 单阶段优化稳定运行；
  - 多阶段循环；
  - MLE 更新；
  - 末阶段自适应长度。

- 尚待后续：
  - 论文 Fig.5/6/8/9 风格的批量实验脚本与统一出图；
  - 多场景 Monte Carlo 统计结果（MSE/CRB 曲线）整理。

## 新增：本地结果保存与离线画图

- `main.py`
  - 新增 `run_method_case(...)`：按方法名与 `eta` 运行并返回可序列化结果。
  - 新增 `save_results_bundle(...)`：将结果保存到本地 `npz/json`。
  - `run_multistage_with_mle(...)` 返回中加入 `stage_histories`，用于后续收敛图绘制。
  - 增加 `if __name__ == "__main__":` 入口保护，便于被其他脚本导入复用。

- `generate_results.py`
  - 新增结果生成脚本，默认运行三种方法：
    - `communication_only (eta=0.0)`
    - `tradeoff (eta=0.5)`
    - `sensing_only (eta=1.0)`
  - 生成并保存：
    - `results/isac_results.npz`
    - `results/isac_results_meta.json`

- `plot_saved_results.py`
  - 新增离线画图脚本，从本地 `results/isac_results.npz` 读取数据。
  - 生成并保存图像，同时 `plt.show()` 弹窗显示：
    - `results/figures/performance_comparison.png`
    - `results/figures/convergence_and_trajectory.png`

## 新增：粗扫描初始化与估计轨迹记录

- `main.py`
  - 新增 `build_coarse_scan_hover_points(...)` 生成初始粗扫描网格点。
  - 新增 `run_initial_coarse_scan(...)`：
    - 模拟粗扫描测距；
    - 通过 MLE 获得初始目标估计；
    - 计算粗扫描能耗。
  - 新增 `compute_scan_energy(...)` 用于粗扫描阶段飞行+悬停能耗结算。
  - `run_multistage_with_mle(...)` 改为自动粗扫描初始化，不再依赖外部手工 `target_hat_init_xy`。
  - 返回结构新增：
    - `target_hat_init_xy`
    - `target_hat_history`
    - `coarse_hover_xy`
    - `scan_energy_used`

- `generate_results.py`
  - 移除手工初始化目标估计输入，统一使用粗扫描自动初始化。

- `plot_saved_results.py`
  - 轨迹图新增：
    - 粗扫描路径；
    - 目标估计更新轨迹（迭代变化）。

- 新增文档：
  - `WORK_LOG.md`（工作日志）
  - `PROJECT_OVERVIEW.md`（项目整体说明：结构、思路、实现、运行方式）
