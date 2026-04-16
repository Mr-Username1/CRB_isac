import numpy as np
import system_model as sysm

# 1. 准备数据
uav_path = np.array([[100, 100], [200, 200], [300, 300]]) # 轨迹坐标
user_pos = np.array([500, 500]) # 用户坐标

cfg = sysm.finalize_config(sysm.SimConfig())
S = np.array([
  [100, 100], [180, 140], [260, 200], [340, 260], [420, 320],
  [500, 380], [580, 440], [660, 500], [740, 560], [820, 620],
], dtype=float)
# 悬停点（由 mu 决定）
Hov = sysm.extract_hover_points(S, cfg.mu)
# 当前目标估计（不是目标真值）
target_hat_xy = np.array([700.0, 900.0])
crb = sysm.crb_xy_sum(Hov, target_hat_xy, cfg)
print("CRB_xt+yt =", crb)