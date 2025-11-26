import numpy as np
import ba

def remove_outlier_points(points3D, obs_list, P_list, corner_list, threshold=5.0):
    """
    删除重投影误差过大的3D点
    points3D    : N x 3 3D点
    obs_list    : [(pt_idx, img_idx, corner_idx), ...]
    P_list      : [P2, P3, P4, ...]
    corner_list : 每张图的角点列表
    threshold   : 重投影误差阈值（像素）
    返回：
        points3D_clean : 删除错误点后的3D点
        obs_clean      : 对应的obs
    """
    N = points3D.shape[0]
    error_sum = np.zeros(N)
    error_count = np.zeros(N)

    # 累积每个点的重投影误差
    for pt_idx, img_idx, corner_idx in obs_list:
        X = points3D[pt_idx]
        P = P_list[img_idx]
        x_obs = corner_list[img_idx][corner_idx]
        x_proj = ba.project_point(P, X)
        err = np.linalg.norm(x_proj - x_obs)
        error_sum[pt_idx] += err
        error_count[pt_idx] += 1

    # 计算每个点平均重投影误差
    mean_error = np.zeros(N)
    mask = error_count > 0
    mean_error[mask] = error_sum[mask] / error_count[mask]

    # 选择误差小于阈值的点
    keep_idx = np.where(mean_error <= threshold)[0]

    # 构建新的points3D和obs
    points3D_clean = points3D[keep_idx]

    # 重建obs索引映射
    idx_map = {old_idx:new_idx for new_idx, old_idx in enumerate(keep_idx)}
    obs_clean = []
    for pt_idx, img_idx, corner_idx in obs_list:
        if pt_idx in idx_map:
            obs_clean.append((idx_map[pt_idx], img_idx, corner_idx))

    print(f"Removed {N - len(keep_idx)} outlier points, remaining: {len(keep_idx)}")
    return points3D_clean, obs_clean
