import numpy as np

# PnP算法
def pnp_linear_dlt(X3D, x2D, K):
    """
    使用线性 DLT 实现 PnP
    X3D: N×3 的 3D 点
    x2D: N×2 的 像素点
    K:   3×3 内参矩阵
    返回:
        R, t    （相机姿态）
    """

    # 像素点归一化到归一化相机坐标
    Kinv = np.linalg.inv(K)
    x_norm = []
    for u, v in x2D:
        ray = Kinv @ np.array([u, v, 1.0])
        x_norm.append(ray[:2] / ray[2])  # (xn, yn)
    x_norm = np.array(x_norm)

    N = len(X3D)
    A = []

    for i in range(N):
        X, Y, Z = X3D[i]
        xn, yn = x_norm[i]

        # 对应 PnP 的 DLT 方程
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -xn*X, -xn*Y, -xn*Z, -xn])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -yn*X, -yn*Y, -yn*Z, -yn])

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    # 分解 P = [R | t]
    R = P[:, :3]
    t = P[:, 3]

    # 正交化 R （确保满足旋转矩阵条件）
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    # 保证旋转矩阵右手系
    if np.linalg.det(R) < 0:
        R *= -1
        t *= -1

    # scale 不重要，仅方向和位置重要
    return R, t

# 图像特征点的3D-2D对应关系
def compute_2D_3D_correspondences(points3d, obs12, matches23):
    """
    points3d: 初始三角化的 3D 点
    obs12: [(i1, i2)] 每个3D点对应图1和图2的角点编号
    matches23: 图2与图3的 inlier 匹配

    返回:
        X3D: 3D 点 (M×3)
        x3:  对应图3的像素点(M×2)
    """

    # 建立 图2角点编号 -> 在点云中对应的3D点编号
    map_2d2_to_3d = {}
    for idx, (i1, i2) in enumerate(obs12):
        map_2d2_to_3d[i2] = idx

    X3D = []
    x3 = []

    for (i2, i3) in matches23:
        if i2 in map_2d2_to_3d:
            j = map_2d2_to_3d[i2]
            X3D.append(points3d[j])
            x3.append(corners3[i3])

    return np.array(X3D), np.array(x3)



def add_new_image(K_new, corners_new, matches_prev_new, sfm):
    """
    加入新图像（例如第3张图）

    输入：
        K_new  : 新图像内参
        corners_new : 新图角点
        matches_prev_new: 上一张图与新图的匹配，例如 matches23
        sfm    : 当前 SfM 状态（包含：points3d, obs, P1, P2, ...）

    输出：
        更新后的 sfm（包含新的点云、P_new、新R,t等）
    """

    # --------------------------------------------------------
    # Step 1：构建 2D-3D 对应关系（关键步骤）
    # --------------------------------------------------------
    points3d = sfm["points3d"]
    obs = sfm["obs"]            # 每个点的 (i_prev) 记录
    corners_prev = sfm["corners_prev"]   # 上一张图的角点
    P_prev = sfm["P_prev"]

    # 建立：上一张图角点编号 → 对应的3D点编号
    map_prev_to_3d = {}
    for p_idx, (i_prev, _) in enumerate(obs):
        map_prev_to_3d[i_prev] = p_idx

    X3D_list = []
    x2D_list = []

    for (i_prev, i_new) in matches_prev_new:
        if i_prev in map_prev_to_3d:
            p_idx = map_prev_to_3d[i_prev]
            X3D_list.append(points3d[p_idx])
            x2D_list.append(corners_new[i_new])

    X3D_list = np.array(X3D_list)
    x2D_list = np.array(x2D_list)

    # --------------------------------------------------------
    # Step 2：使用自写 PnP 求新图姿态
    # --------------------------------------------------------
    R_new, t_new = pnp_linear_dlt(X3D_list, x2D_list, K_new)
    P_new = K_new @ np.hstack([R_new, t_new.reshape(3,1)])

    # --------------------------------------------------------
    # Step 3：三角化新的点（System Grow）
    # --------------------------------------------------------
    new_points = []
    new_obs = []

    for (i_prev, i_new) in matches_prev_new:
        X = triangulate_point(P_prev, P_new, corners_prev[i_prev], corners_new[i_new])
        new_points.append(X)
        new_obs.append((i_prev, i_new))

    new_points = np.array(new_points)

    # --------------------------------------------------------
    # Step 4：合并点云与观测
    # --------------------------------------------------------
    sfm["points3d"] = np.vstack([sfm["points3d"], new_points])
    sfm["obs"]      = sfm["obs"] + new_obs

    # --------------------------------------------------------
    # Step 5：记录新相机状态
    # --------------------------------------------------------
    sfm["P_new"] = P_new
    sfm["R_new"] = R_new
    sfm["t_new"] = t_new
    sfm["corners_prev"] = corners_new
    sfm["P_prev"] = P_new

    return sfm
