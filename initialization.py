# 进行初始化

import numpy as np
import triangulation

# 读取相机文件txt中的内参外参
def load_cam_file(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f]

    # 找到 extrinsic 部分
    e_idx = lines.index("extrinsic") + 1
    Rt = []
    for i in range(4):
        Rt.append(list(map(float, lines[e_idx + i].split())))
    Rt = np.array(Rt)

    R = Rt[:3, :3]
    t = Rt[:3, 3]

    # 找到 intrinsic 部分
    i_idx = lines.index("intrinsic") + 1
    K = []
    for i in range(3):
        K.append(list(map(float, lines[i_idx + i].split())))
    K = np.array(K)

    return K, R, t
'''
def initialize_with_two_views(K, corners1, corners2, matches12, F12):
    """
    使用两张图进行初始化，返回：
        P1, P2   : 投影矩阵
        R2, t2   : 第二个相机姿态
        points3d : 三角化出的点云
        obs      : 每个点的观测匹配关系 (i1, i2)
    """

    # ------------------------------------------------
    # Step 1: F -> E
    # ------------------------------------------------
    E = K.T @ F12 @ K

    # ------------------------------------------------
    # Step 2: 分解 E → 4个候选 (R,t)
    # ------------------------------------------------
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:,2]

    candidates = [
        (R1,  t),
        (R1, -t),
        (R2,  t),
        (R2, -t)
    ]

    # ------------------------------------------------
    # Step 3: 使用正深度约束选出正确姿态
    # ------------------------------------------------
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])

    best_infront = -1
    best_pose = None

    for R, t in candidates:
        P2 = K @ np.hstack([R, t.reshape(3,1)])
        count = 0
        for (i,j) in matches12[:50]:     # 检查 50 个点即可
            X = triangulation.triangulate_point(P1, P2, corners1[i], corners2[j])
            z1 = X[2]
            z2 = (R @ X + t)[2]
            if z1 > 0 and z2 > 0:
                count += 1

        if count > best_infront:
            best_infront = count
            best_pose = (R, t)

    R2, t2 = best_pose
    P2 = K @ np.hstack([R2, t2.reshape(3,1)])

    # ------------------------------------------------
    # Step 4: 对所有匹配进行三角化
    # ------------------------------------------------
    points3d = []
    obs = []

    for (i1, i2) in matches12:
        X = triangulation.triangulate_point(P1, P2, corners1[i1], corners2[i2])
        points3d.append(X)
        obs.append((i1, i2))

    return P1, P2, R2, t2, np.array(points3d), obs
'''
import numpy as np

def decompose_essential(E):
    """
    从本征矩阵E分解得到R,t四种可能组合
    """
    U, S, Vt = np.linalg.svd(E)
    
    # 保证旋转矩阵的行列式为+1
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = U[:, 2]
    t2 = -U[:, 2]

    poses = [
        (R1, t1),
        (R1, t2),
        (R2, t1),
        (R2, t2)
    ]
    return poses

def select_correct_pose(P1, matches, corners2, corners3, K2, K3, poses):
    """
    使用正深度约束选择正确的R,t
    """
    max_positive = 0
    best_pose = None
    for R, t in poses:
        P2 = K3 @ np.hstack((R, t.reshape(3,1)))
        positive_count = 0
        for m in matches:
            idx2, idx3 = m
            x1 = corners2[idx2]
            x2 = corners3[idx3]
            X = triangulation.triangulate_point(P1, P2, x1, x2)
            # 检查X在两相机前方
            X_cam1 = P1 @ np.hstack((X,1))
            X_cam2 = P2 @ np.hstack((X,1))
            if X_cam1[2] > 0 and X_cam2[2] > 0:
                positive_count += 1
        if positive_count > max_positive:
            max_positive = positive_count
            best_pose = (R, t)
    return best_pose

def initialize_two_view(corners2, corners3, inlier_matches23, K2, K3, F23):
    """
    初始化函数: 返回投影矩阵P2, P3以及初始3D点
    输入:
        corners2, corners3        : 两张图的角点列表
        inlier_matches23          : 几何验证后的可靠匹配 [(idx2, idx3), ...]
        K2, K3                    : 两张图的内参
        F23                        : 两张图的基础矩阵
    输出:
        P2, P3, points3D
    """
    # 设置图2为世界坐标系
    P1 = K2 @ np.hstack((np.eye(3), np.zeros((3,1))))

    # 计算本征矩阵E
    E = K3.T @ F23 @ K2

    poses = decompose_essential(E)
    R, t = select_correct_pose(P1, inlier_matches23, corners2, corners3, K2, K3, poses)

    # 构建图3的投影矩阵
    P2 = K3 @ np.hstack((R, t.reshape(3,1)))

    # 三角化所有匹配点
    points3D = []
    for m in inlier_matches23:
        idx2, idx3 = m
        x1 = corners2[idx2]
        x2 = corners3[idx3]
        X = triangulation.triangulate_point(P1, P2, x1, x2)
        points3D.append(X)
    points3D = np.array(points3D)

    return P1, P2, points3D

