'''
用于生成阶段1要求的特征匹配结果：
    每两张图像的匹配结果（匹配特征点连线）
    任选匹配点，可视化核线
'''

import cv2
import numpy as np

import harris
import match 
import geo_veri
import draw1
print(draw1.__file__)

# 读取每张图
img2 = cv2.imread("images/00000022.jpg")
img3 = cv2.imread("images/00000023.jpg")
img4 = cv2.imread("images/00000024.jpg")

# 寻找每张图的harris角点，并得到描述子
corners2, desc2 = harris.harris_features(img2)
corners3, desc3 = harris.harris_features(img3)
corners4, desc4 = harris.harris_features(img4)

# 特征匹配，找到各自对应的角点
matches23 = match.bf_match(desc2, desc3)
matches34 = match.bf_match(desc3, desc4)
matches24 = match.bf_match(desc2, desc4)

# 几何验证
inlier_matches23, F23 = geo_veri.ransac_fundamental(corners2, corners3, matches23, thresh=1.0, max_iters=500)
inlier_matches34, F34 = geo_veri.ransac_fundamental(corners3, corners4, matches34, thresh=1.0, max_iters=500)
inlier_matches24, F24 = geo_veri.ransac_fundamental(corners2, corners4, matches24, thresh=1.0, max_iters=500)

# 进行画图：特征点连线
match_img_23 = draw1.draw_matches(img2, img3, corners2, corners3, inlier_matches23)
cv2.imwrite("outputs/match_23.jpg", match_img_23)

match_img_34 = draw1.draw_matches(img3, img4, corners3, corners4, inlier_matches34)
cv2.imwrite("outputs/match_34.jpg", match_img_34)

match_img_24 = draw1.draw_matches(img2, img4, corners2, corners4, inlier_matches24)
cv2.imwrite("outputs/match_24.jpg", match_img_24)