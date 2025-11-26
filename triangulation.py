import numpy as np

# 第一次三角化
def triangulate_point(P1, P2, x1, x2):
    """
    使用DLT算法三角化单个点
    输入:
        P1, P2 : 3×4 投影矩阵
        x1, x2 : 两张图像中的对应像素坐标 (x, y)
    输出:
        X : 3D点 (3,)
    """

    x1, y1 = x1
    x2, y2 = x2

    A = np.zeros((4,4))
    A[0] = x1 * P1[2] - P1[0]
    A[1] = y1 * P1[2] - P1[1]
    A[2] = x2 * P2[2] - P2[0]
    A[3] = y2 * P2[2] - P2[1]

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X /= X[3]

    return X[:3]

