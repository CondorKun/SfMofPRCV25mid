import numpy as np
import triangulation
# -------------------------------
# 投影函数
# -------------------------------
def project_point(P, X):
    X_h = np.hstack([X,1])
    x = P @ X_h
    return x[:2] / x[2]

# -------------------------------
# 纯 NumPy BA 优化
# -------------------------------
def bundle_adjustment(P_list, points3D, obs, corner_list, max_iters=10, lr=1e-6):
    points3D = points3D.copy()
    P_list = [P.copy() for P in P_list]

    for it in range(max_iters):
        total_error = 0
        for pt_idx, img_idx, corner_idx in obs:
            X = points3D[pt_idx]
            P = P_list[img_idx]
            x_obs = corner_list[img_idx][corner_idx]
            x_proj = project_point(P, X)
            error = x_proj - x_obs
            total_error += np.sum(error**2)

            # 对3D点梯度下降
            grad_X = np.zeros(3)
            eps = 1e-6
            for i in range(3):
                X_eps = X.copy()
                X_eps[i] += eps
                x_proj_eps = project_point(P, X_eps)
                grad_X[i] = np.sum((x_proj_eps - x_proj) * error) / eps
            points3D[pt_idx] -= lr * grad_X

            # 对相机平移梯度下降
            t = P[:,3].copy()
            grad_t = np.zeros(3)
            for i in range(3):
                t_eps = t.copy()
                t_eps[i] += eps
                P_eps = P.copy()
                P_eps[:,3] = t_eps
                x_proj_eps = project_point(P_eps, X)
                grad_t[i] = np.sum((x_proj_eps - x_proj) * error) / eps
            P_list[img_idx][:,3] -= lr * grad_t

        print(f"Iteration {it+1}/{max_iters}, total reprojection error: {total_error:.2f}")

    return P_list, points3D

# -------------------------------
# 构造 obs 的安全函数
# -------------------------------
def construct_obs_safe(inlier_matches23, inlier_matches34, points3D, corners2, corners3, corners4, P_list):
    """
    动态生成obs，保证索引不会越界
    """
    obs_list = []

    # 初始点 2-3 图
    for pt_idx, (i2, i3) in enumerate(inlier_matches23):
        obs_list.append((pt_idx, 0, i2))  # 图2
        obs_list.append((pt_idx, 1, i3))  # 图3

    # 新点 3-4 图
    new_point_indices = []
    for idx, (idx3, idx4) in enumerate(inlier_matches34):
        X = triangulation.triangulate_point(P_list[1], P_list[2], corners3[idx3], corners4[idx4])
        points3D = np.vstack([points3D, X])
        pt_idx = points3D.shape[0] - 1
        new_point_indices.append((pt_idx, idx3, idx4))

    for pt_idx, idx3, idx4 in new_point_indices:
        obs_list.append((pt_idx, 1, idx3))  # 图3
        obs_list.append((pt_idx, 2, idx4))  # 图4

    return obs_list, points3D
