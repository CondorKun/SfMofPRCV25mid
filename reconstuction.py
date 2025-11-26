import numpy as np

# -------------------------------
# 投影函数
# -------------------------------
def project_point(P, X):
    X_h = np.hstack([X,1])
    x = P @ X_h
    return x[:2] / x[2]

# -------------------------------
# 保存点云和位姿函数
# -------------------------------
def save_reconstruction(points3D, P_list, obs_list, corner_list, img_list, K_list, R_true_list, t_true_list,
                        point_txt="points.txt", pose_txt="poses.txt", error_txt="errors.txt"):
    """
    points3D      : N x 3
    P_list        : 优化后的投影矩阵列表 [P2,P3,P4]
    obs_list      : [(pt_idx, img_idx, corner_idx), ...]
    corner_list   : 每张图的角点 [(x,y), ...]
    img_list      : 每张图的原始图像数组，用于提取颜色
    K_list        : 内参列表
    R_true_list   : 真值旋转矩阵列表
    t_true_list   : 真值平移向量列表
    """
    N = points3D.shape[0]
    M = len(P_list)

    # -------------------------------
    # 1. 保存带颜色的点云
    # -------------------------------
    point_lines = []
    for i, X in enumerate(points3D):
        # 取第一个观测图的颜色
        color = np.array([128,128,128])  # 默认灰色
        for pt_idx, img_idx, corner_idx in obs_list:
            if pt_idx == i:
                x, y = corner_list[img_idx][corner_idx]
                x = int(round(x))
                y = int(round(y))
                h, w, _ = img_list[img_idx].shape
                if 0 <= x < w and 0 <= y < h:
                    color = img_list[img_idx][y, x]
                break
        line = f"{i} {X[0]} {X[1]} {X[2]} {color[0]} {color[1]} {color[2]}"
        point_lines.append(line)
    with open(point_txt, "w") as f:
        f.write("\n".join(point_lines))
    print(f"Saved {N} 3D points with color to {point_txt}")

    # -------------------------------
    # 2. 保存相机外参 (4x4)
    # -------------------------------
    pose_lines = []
    for i, P in enumerate(P_list):
        # 反解 [R|t] from P = K[R|t] (近似)
        K_inv = np.linalg.inv(K_list[i])
        Rt = K_inv @ P
        R = Rt[:,:3]
        t = Rt[:,3]
        # 生成4x4矩阵
        pose_mat = np.eye(4)
        pose_mat[:3,:3] = R
        pose_mat[:3,3] = t
        # 保存为一行文本
        pose_line = " ".join(map(str, pose_mat.flatten()))
        pose_lines.append(pose_line)
    with open(pose_txt, "w") as f:
        f.write("\n".join(pose_lines))
    print(f"Saved {M} camera poses to {pose_txt}")

    # -------------------------------
    # 3. 计算重投影误差
    # -------------------------------
    reproj_errors = []
    for i in range(M):
        pts_err = []
        for pt_idx, img_idx, corner_idx in obs_list:
            if img_idx != i:
                continue
            X = points3D[pt_idx]
            P = P_list[i]
            x_obs = corner_list[i][corner_idx]
            x_proj = project_point(P, X)
            err = np.linalg.norm(x_proj - x_obs)
            pts_err.append(err)
        mean_err = np.mean(pts_err) if pts_err else 0
        reproj_errors.append(mean_err)
        print(f"Image {i} mean reprojection error: {mean_err:.3f}")

    # -------------------------------
    # 4. 计算位姿误差
    # -------------------------------
    pose_errors = []
    for i in range(M):
        for j in range(i+1,M):
            # 计算旋转误差
            R_est = np.linalg.inv(P_list[i][:,:3]) @ P_list[j][:,:3]
            R_true = R_true_list[i].T @ R_true_list[j]
            # 旋转误差角度
            cos_theta = (np.trace(R_est.T @ R_true)-1)/2
            cos_theta = np.clip(cos_theta,-1,1)
            rot_err = np.arccos(cos_theta) * 180/np.pi  # 度
            # 平移误差角度
            t_est = P_list[j][:,3] - P_list[i][:,3]
            t_true = t_true_list[j] - t_true_list[i]
            t_err = np.arccos(np.clip(np.dot(t_est,t_true)/(np.linalg.norm(t_est)*np.linalg.norm(t_true)), -1,1)) * 180/np.pi
            pose_errors.append((i,j,rot_err,t_err))
            print(f"Pose error between image {i} and {j}: rotation {rot_err:.2f} deg, translation {t_err:.2f} deg")

    # -------------------------------
    # 5. 保存统计信息
    # -------------------------------
    with open(error_txt,"w") as f:
        f.write(f"Number of 3D points: {N}\n")
        for i, err in enumerate(reproj_errors):
            f.write(f"Image {i} mean reprojection error: {err}\n")
        for i,j,rot_err,t_err in pose_errors:
            f.write(f"Pose error image {i}-{j}: rotation {rot_err} deg, translation {t_err} deg\n")
    print(f"Saved reprojection and pose errors to {error_txt}")
