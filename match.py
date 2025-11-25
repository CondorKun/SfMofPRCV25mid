"""
该程序进行 Brute-Force 匹配描述子 (L2距离) + 比率测试
    
    输入：
        desc1: numpy array, 图1描述子, shape = (N1, D)
        desc2: numpy array, 图2描述子, shape = (N2, D)
        ratio_thresh: Lowe比率测试阈值, 默认0.8
    
    输出：
        matches: list of tuples (i, j)
            i -> desc1索引, j -> desc2索引
"""

import numpy as np

def bf_match(desc1, desc2, ratio_thresh=0.8):
    
    matches = []
    
    for i, d1 in enumerate(desc1):
        # 计算图1的第i个描述子到图2所有描述子的欧氏距离
        distances = np.linalg.norm(desc2 - d1, axis=1)
        if len(distances) < 2:
            continue
        # 找到最近两个距离
        idx_sorted = np.argsort(distances)
        nearest, second_nearest = distances[idx_sorted[0]], distances[idx_sorted[1]]
        # 比率测试
        if nearest / second_nearest < ratio_thresh:
            matches.append((i, idx_sorted[0]))

    print("get match")
    
    return matches
