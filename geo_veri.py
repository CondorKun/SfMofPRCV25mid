'''
几何验证

'''

import numpy as np
'''
def normalize_points(pts):
    #归一化点坐标，使平均距离sqrt(2)

    pts = np.array(pts)
    mean = pts.mean(axis=0)
    std = pts.std(axis=0)
    s = np.sqrt(2) / std
    T = np.array([[s, 0, -s*mean[0]],
                  [0, s, -s*mean[1]],
                  [0, 0, 1]])
    pts_h = np.hstack([pts, np.ones((pts.shape[0],1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm, T
    '''
def normalize_points(pts):
    pts = np.asarray(pts)

    # 正确的形状应为 (N,2)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"pts must be of shape (N,2), got {pts.shape}")

    # 计算均值
    mean = np.mean(pts, axis=0)  # (2,)
    pts_centered = pts - mean

    # 计算平均距离（标量）
    mean_dist = np.mean(np.sqrt(np.sum(pts_centered**2, axis=1)))

    # 比例 s —— 标量！
    s = np.sqrt(2) / mean_dist

    T = np.array([
        [s, 0, -s * mean[0]],
        [0, s, -s * mean[1]],
        [0, 0, 1]
    ])

    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_h.T).T[:, :2]

    return pts_norm, T


def eight_point_F(pts1, pts2):
    #8点法计算基础矩阵

    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)
    
    A = []
    for p1, p2 in zip(pts1_norm, pts2_norm):
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
    A = np.array(A)
    
    # SVD求解F
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3,3)
    
    # 强制F秩为2
    Uf, Sf, Vtf = np.linalg.svd(F)
    Sf[2] = 0
    F_rank2 = Uf @ np.diag(Sf) @ Vtf
    
    # 反归一化
    F_final = T2.T @ F_rank2 @ T1
    return F_final / F_final[2,2]

def ransac_fundamental(corners1, corners2, matches, thresh=0.01, max_iters=1000):
    """
    纯Python RANSAC几何验证
    
    Args:
        corners1, corners2: 角点坐标列表
        matches: 匹配索引对
        thresh: 核线距离阈值 (单位为像素)
        max_iters: RANSAC迭代次数
    
    Returns:
        best_inliers: 内点匹配索引对
        best_F: 对应基础矩阵
    """
    if len(matches) < 8:
        return [], None
    
    pts1 = np.array([corners1[i] for i, j in matches])
    pts2 = np.array([corners2[j] for i, j in matches])
    
    best_inliers = []
    best_F = None
    
    for _ in range(max_iters):
        # 随机选择8个点
        idx = np.random.choice(len(matches), 8, replace=False)
        F_candidate = eight_point_F(pts1[idx], pts2[idx])
        
        # 计算所有点到核线的距离
        pts1_h = np.hstack([pts1, np.ones((pts1.shape[0],1))])
        pts2_h = np.hstack([pts2, np.ones((pts2.shape[0],1))])
        
        # 点到核线距离公式: d = |x2^T F x1| / sqrt(a^2 + b^2)
        Fx1 = F_candidate @ pts1_h.T  # shape 3xN
        Ftx2 = F_candidate.T @ pts2_h.T
        
        d = np.abs(np.sum(pts2_h.T * Fx1, axis=0)) / (Fx1[0,:]**2 + Fx1[1,:]**2)**0.5
        inliers_idx = np.where(d < thresh)[0]
        
        if len(inliers_idx) > len(best_inliers):
            best_inliers = inliers_idx
            best_F = F_candidate
    
    # 返回匹配索引对
    best_inliers_matches = [matches[i] for i in best_inliers]
    print("get veri")
    return best_inliers_matches, best_F
