# 
# 
'''
该程序用于进行harris角点检测，并生成简单描述子

输入：
    image: 输入彩色图像 (BGR)
    window_size: Harris响应窗口大小
    descriptor_size: 描述子窗口大小 (必须是奇数，保证角点在窗口中心)
    k: Harris常数
    threshold_ratio: 响应阈值比例 (R.max() * threshold_ratio)

    后面四项都有缺省值，所以秩只需指定图片

输出：
    corners: list of (x, y) 坐标
    descriptors: numpy array, 每行对应一个角点的描述子
'''

import cv2
import numpy as np

def harris_features(image, window_size=3, descriptor_size=5, k=0.04, threshold_ratio=0.01):
    
    # 处理成灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # 计算图像梯度
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # 卷积求和
    kernel = np.ones((window_size, window_size))
    Sxx = cv2.filter2D(Ixx, -1, kernel)
    Syy = cv2.filter2D(Iyy, -1, kernel)
    Sxy = cv2.filter2D(Ixy, -1, kernel)

    # Harris响应
    R = (Sxx * Syy - Sxy ** 2) - k * (Sxx + Syy) ** 2

    # 阈值 + 非极大值
    threshold = threshold_ratio * R.max()
    corners_idx = np.argwhere(R > threshold)
    corners = [tuple(pt[::-1]) for pt in corners_idx]  # (x, y)

    # 描述子
    offset = descriptor_size // 2
    padded = np.pad(gray, ((offset, offset), (offset, offset)), mode='constant')
    descriptors = []
    for x, y in corners:
        patch = padded[y:y+descriptor_size, x:x+descriptor_size].flatten()
        norm = np.linalg.norm(patch)
        if norm > 0:
            patch = patch / norm
        descriptors.append(patch)
    descriptors = np.array(descriptors)

    print("get harris")

    return corners, descriptors
