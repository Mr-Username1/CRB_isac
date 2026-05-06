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

---

日期：2026-05-06

## 本轮目标

- 用**序贯测距 EKF**替代（或与）网格 MLE 做目标水平位置估计；
- 预测用上一时刻后验、观测雅可比与 \(h\) 在**当前先验** \(x_{k|k-1}\) 处计算；
- 粗扫 MLE 初值 + 大对角 \(P_0\)；`generate_results.py` 中 `scenario_name` 与 `localizer` 可切换。

## 新增 / 修改文件

- **`ekf_range_localization.py`**（新）
  - `StaticRangeEKF2D`：静态 \(F=I\)，小 \(Q\)，Joseph 形式协方差更新；
  - `default_prior_variance_xy` / `default_process_variance`：与地图尺度 \(\min(L_x,L_y)\) 成比例。
- **`simulation_pipeline.py`**
  - `run_multistage_with_mle(..., localizer="ekf"|"mle")`：EKF 时对**本阶段每个新悬停点**按行序 `update_one`；MLE 分支保持原 `mle_grid_search`+`ref_xy`；
  - `run_method_case(..., localizer=...)` 与返回字段 `localizer`；
  - `stage_logs` 增加 `localizer`。
- **`generate_results.py`**：顶部 `scenario_name`、`localizer` 两个开关；打印二者。
- **`io_utils.py`**：`meta.json` 写入 `localizer`。
- **`main.py`**：示例调用显式 `localizer="ekf"` 并打印。

## EKF 关键公式（与代码一致）

**状态** \(\mathbf{x}=[x_t,y_t]^\top\)。**量测** \(z_k\) 为斜距，UAV 水平位置 \(\mathbf{u}_k=[x_k^u,y_k^u]^\top\)。

- **预测**（静态）：\(\mathbf{x}_{k|k-1}=\mathbf{x}_{k-1|k-1}\)，\(\mathbf{P}_{k|k-1}=\mathbf{P}_{k-1|k-1}+\mathbf{Q}\)，\(\mathbf{Q}=q\mathbf{I}\)，\(q\) 见参数表。
- **预测距离**（在 \(\mathbf{x}_{k|k-1}\) 处）：\(d_s=\sqrt{H_{\mathrm{est}}^2+\|\mathbf{u}_k-\mathbf{x}_{k|k-1}\|^2}\)，\(h(\mathbf{x}_{k|k-1})=d_s\)。
- **雅可比**（1×2）：\(\mathbf{H}_k=\big[-(x_k^u-x^-)/d_s,\;-(y_k^u-y^-)/d_s\big]\)。
- **量测噪声方差**（与 `system_model.sigma2_measurement_from_g` 一致，在 \(d_s\) 处算 \(g\)）：\(R_k = a\sigma_0^2/(P_w G_p g(d_s))\)（下限截断 \(10^{-12}\)）。\(H_{\mathrm{est}}=H+\) `model_mismatch_h`，\(\beta_{0,\mathrm{est}}=\beta_0\cdot10^{\mathrm{model\_mismatch\_beta0\_db}/10}\)（与当前 MLE 假设对齐）。
- **更新**：\(\mathbf{K}_k=\mathbf{P}_{k|k-1}\mathbf{H}_k^\top(S_k)^{-1}\)，\(S_k=\mathbf{H}_k\mathbf{P}_{k|k-1}\mathbf{H}_k^\top+R_k\)；\(\mathbf{x}_{k|k}=\mathbf{x}_{k|k-1}+\mathbf{K}_k(z_k-h)\)；Joseph：\(\mathbf{P}_{k|k}=(\mathbf{I}-\mathbf{K}_k\mathbf{H}_k)\mathbf{P}_{k|k-1}(\mathbf{I}-\mathbf{K}_k\mathbf{H}_k)^\top+\mathbf{K}_k R_k\mathbf{K}_k^\top\)（再对称化）。

**初值**：均值 \(\mathbf{x}_{0|0}\) = 粗扫网格 MLE（`run_initial_coarse_scan` 内 `mle_grid_search`）；**不**将粗扫量测再次序贯送入 EKF，避免与初值同源双重计数。

## 参数表（EKF 专用常数，与 `SimConfig` 几何联动）

| 符号 | 代码位置 | 取值 / 规则 | 说明 |
|------|-----------|-------------|------|
| \(\sigma_0^2\)（每轴先验方差） | `default_prior_variance_xy` | \((0.32\cdot\min(L_x,L_y))^2\) | 大对角 \(P_0=\sigma_0^2 I\) |
| \(q\)（每轴过程方差一步） | `default_process_variance` | \(\max\big((1.5\times10^{-4}\cdot\min(L_x,L_y))^2,\;10^{-4}\big)\) | 静态目标，仅防奇异 |
| `prior_frac` | `StaticRangeEKF2D(..., prior_frac=0.32)` | 默认 0.32 | 可调大/小先验 |
| \(H_{\mathrm{est}},\beta_{0,\mathrm{est}}\) | `_predicted_range_and_H_and_R` | 同 `simulation_pipeline.mle_grid_search` | 与现有 MLE 噪声模型一致 |
| Joseph | `update_one` | 对称化 \(\tfrac12(\mathbf{P}+\mathbf{P}^\top)\) | 数值稳定 |

## 使用说明

- `generate_results.py` 顶部：`scenario_name = "..."`，`localizer = "ekf"` 或 `"mle"`。
- 若需旧行为全文网格 MLE：设 `localizer="mle"`。

## 备注

- EKF 仅替换**目标位置估计**；轨迹优化仍用 \(\widetilde{\mathrm{CRB}}(\hat{\mathbf{p}})\) 与 SCA，与论文结构一致。
- 粗扫初值仍用 MLE（小区域粗网格），与「初值来自粗扫」要求一致。

### 问题修复（同日）

- `ekf_range_localization.StaticRangeEKF2D.update_one`：`H @ P_minus @ H.T` 为 shape `(1,1)`，直接 `float(...)` 在部分 NumPy 下触发 `TypeError`；改为对 `S_mat` 做 `reshape(-1)[0]` 再与 `R` 相加得到标量 \(S_k\)。
- 本地已跑通：`python generate_results.py`（`paper_baseline` + `ekf` 三方法）、`localizer='mle'` 单 case、以及 200 次随机 `update_one` 压力脚本。
