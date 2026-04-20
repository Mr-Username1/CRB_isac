# 工作日志

日期：2026-04-16

## 本轮目标

- 去掉手工指定 `target_hat_init_xy` 的非真实场景设定。
- 新增“初始粗扫描阶段”用于目标粗定位。
- 在多阶段迭代中保存每次目标估计更新轨迹。
- 同步更新结果保存与离线画图脚本。

## 本轮完成项

- 在 `main.py` 新增粗扫描相关函数：
  - `build_coarse_scan_hover_points(...)`
  - `compute_scan_energy(...)`
  - `run_initial_coarse_scan(...)`
- 在 `run_multistage_with_mle(...)` 接入：
  - 粗扫描观测与初始估计；
  - 粗扫描能耗扣减；
  - 粗扫描悬停点并入历史悬停集合；
  - 每阶段 MLE 更新后记录 `target_hat_history`。
- `run_method_case(...)` 和 `save_results_bundle(...)` 同步新数据结构：
  - 增加 `target_hat_init_xy`、`target_hat_history`、`scan_energy_used`、`coarse_hover_xy`。
- 更新 `generate_results.py`：
  - 移除外部手工 `target_hat_init_xy` 输入，改为内部粗扫描自动估计。
- 更新 `plot_saved_results.py`：
  - 轨迹图增加粗扫描路径；
  - 增加目标估计更新轨迹（从初始估计到各阶段更新）。

## 结果验证

- 重新运行：
  - `python generate_results.py`
  - `python plot_saved_results.py`
- 确认本地结果与图像正常保存，并且图像可弹窗显示。

## 备注

- 当前估计更新是“阶段级”更新（每阶段结束后一次 MLE），与论文多阶段流程一致。
- 若后续需要“每个 SCA 子迭代内更新目标估计”，可再扩展为更细粒度日志与估计流程。

---

日期：2026-04-20

## 本轮目标

- 按论文口径修正此前简化项（重点：P2(m) 的累计通信项与步长搜索方式）。
- 提供可选仿真场景参数，并支持更贴近现实的高噪声环境。

## 本轮完成项

- `system_model.py`
  - 在 `SimConfig` 中新增场景与噪声相关参数：
    - `scenario_name`, `gp_scale`
    - `nlos_bias_mean/std`
    - `outlier_prob/std`
    - `model_mismatch_h`, `model_mismatch_beta0_db`
  - 新增 `apply_scenario_preset(...)`，提供三种可选场景：
    - `paper_baseline`
    - `high_noise_realistic`
    - `extreme_noise`
  - `finalize_config(...)` 改为先应用场景预设，再计算派生参数（`Gp = gp_scale * B`）。

- `p2_solver.py`
  - `StageData` 新增：
    - `N_prev_total`
    - `R_prev_sum`
  - 目标函数中的通信项按论文 Eq.(33) 口径改为“累计平均速率”：
    - 当前阶段线性化速率按累计分母缩放；
    - 结合历史 `R_prev_sum` 与 `N_prev_total`。
  - 新增 `_rate_value(...)` 用于真实目标值评估（累计速率口径）。
  - 步长更新从“回溯接受首个可行点”改为“候选集线搜索择优”：
    - `line_search_candidates = (1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05)`
    - 沿下降方向选择目标值最小的 `omega`。

- `main.py`
  - 新增 `configure_simulation(...)`，可在运行前切换场景参数。
  - `simulate_range_measurements(...)` 接入更现实噪声项：
    - 高斯测距噪声
    - NLOS 正偏置
    - 稀疏异常点扰动
  - `mle_grid_search(...)` 接入模型失配（高度与 `beta0` 偏差）。
  - 多阶段循环中加入累计通信统计：
    - 每阶段更新 `R_prev_sum` 和 `N_prev_total`
    - 并传入 `StageData`。
  - 结果元数据新增场景与噪声参数记录。

- `generate_results.py`
  - 运行入口新增 `scenario_name` 选择；
  - 开始仿真前调用 `configure_simulation(...)` 切换场景（默认高噪声现实场景）。

## 结果说明

- 目前已支持“论文基线/高噪声现实/极端噪声”三套参数快速切换。
- 优化目标的通信项已改为累计口径，更贴近论文分阶段定义。

---

日期：2026-04-20（结构重构）

## 本轮目标

- 整合 `main.py` 与 `generate_results.py` 的职责重叠；
- 清理项目中冗余/不清晰结构，形成更清晰的模块边界。

## 本轮完成项

- 新增 `simulation_pipeline.py`
  - 从 `main.py` 抽离仿真核心流程：
    - 量测仿真
    - MLE 网格搜索
    - 粗扫描初始化
    - 多阶段主循环
    - 单方法运行接口 `run_method_case(...)`

- 新增 `config_factory.py`
  - 统一配置构建入口：
    - `build_sim_config(...)`
    - `build_energy_config(...)`
    - `build_solver_config(...)`
    - `build_default_configs(...)`

- 新增 `io_utils.py`
  - 抽离结果保存逻辑 `save_results_bundle(...)`；
  - 提供可选开关 `save_full_trajectories`，便于控制结果冗余。

- 重写 `main.py`
  - 由“全能脚本”改为“轻量单案例演示入口”；
  - 不再维护全局可变配置状态；
  - 仅负责 smoke test。

- 更新 `generate_results.py`
  - 改为调用 `config_factory + simulation_pipeline + io_utils`；
  - 成为推荐的唯一批量实验入口。

- 其他清理
  - `p2_solver.py` 去除误导性头注释和未使用导入；
  - `PROJECT_OVERVIEW.md` 更新为重构后结构与使用方法。

## 本轮收益

- 模块职责清晰：模型/求解/流程/配置/IO 分层明确；
- 降低全局状态副作用，后续扩展 Monte Carlo 并行更安全；
- 入口职责统一，维护成本下降。
