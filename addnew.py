import numpy as np
import triangulation

def project_points_np(X, K, R, t):
    """ 将3D点投影到图像平面 """
    X_h = np.hstack([X, np.ones((X.shape[0],1))])  # N x 4
    P = K @ np.hstack([R, t.reshape(3,1)])         # 3x4
    x_proj = (P @ X_h.T).T                         # N x 3
    x_proj = x_proj[:,:2] / x_proj[:,2:3]
    return x_proj

def pnp_pure_np(X, x_obs, K, iterations=100, lr=1e-6):
    """
    纯NumPy PnP位姿估计
    X: N x 3 3D点
    x_obs: N x 2 对应2D点
    K: 内参
    返回: R, t
    """
    # 初始化旋转向量 r 和平移 t
    r = np.zeros(3)
    t = np.zeros(3)

    for it in range(iterations):
        theta = np.linalg.norm(r)
        if theta < 1e-8:
            R = np.eye(3)
        else:
            k = r / theta
            Kx = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
            R = np.eye(3) + np.sin(theta)*Kx + (1-np.cos(theta))*(Kx@Kx)

        x_proj = project_points_np(X, K, R, t)
        error = x_proj - x_obs  # N x 2

        # 简单梯度下降，数值梯度
        grad_r = np.zeros(3)
        grad_t = np.zeros(3)
        eps = 1e-6
        for i in range(3):
            dr = np.zeros(3); dr[i] = eps
            theta_dr = np.linalg.norm(r + dr)
            if theta_dr < 1e-8:
                R_dr = np.eye(3)
            else:
                k = (r+dr)/theta_dr
                Kx = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
                R_dr = np.eye(3) + np.sin(theta_dr)*Kx + (1-np.cos(theta_dr))*(Kx@Kx)
            x_proj_dr = project_points_np(X, K, R_dr, t)
            grad_r[i] = np.sum((x_proj_dr - x_proj) * error) / eps

            dt = np.zeros(3); dt[i] = eps
            x_proj_dt = project_points_np(X, K, R, t+dt)
            grad_t[i] = np.sum((x_proj_dt - x_proj) * error) / eps

        # 更新
        r -= lr * grad_r
        t -= lr * grad_t

    theta = np.linalg.norm(r)
    if theta < 1e-8:
        R_final = np.eye(3)
    else:
        k = r / theta
        Kx = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
        R_final = np.eye(3) + np.sin(theta)*Kx + (1-np.cos(theta))*(Kx@Kx)
    return R_final, t

def add_new_image_np(P_list, points3D, obs, corners_new, inlier_matches, K_new, corners_prev):
    """
    纯 NumPy 版本加入新图
    """
    # Step 1: 找2D-3D对应
    X_corr = []
    x_new = []
    match_indices = []

    for idx_prev, idx_new in inlier_matches:
        if idx_prev < len(points3D):
            X_corr.append(points3D[idx_prev])
            x_new.append(corners_new[idx_new])
            match_indices.append(idx_prev)

    X_corr = np.array(X_corr)
    x_new = np.array(x_new)

    # Step 2: 纯NumPy PnP求新图位姿
    R_new, t_new = pnp_pure_np(X_corr, x_new, K_new)

    P_new = K_new @ np.hstack([R_new, t_new.reshape(3,1)])

    # Step 3: 三角化新点
    for idx_prev, idx_new in inlier_matches:
        if idx_prev >= len(points3D):
            X = triangulation.triangulate_point(P_list[0], P_new, corners_prev[idx_prev], corners_new[idx_new])
            points3D = np.vstack([points3D, X])
            obs.append((idx_prev, idx_new))

    P_list.append(P_new)
    return P_new, points3D, obs
