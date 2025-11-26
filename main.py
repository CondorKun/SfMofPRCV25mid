import cv2
import numpy as np

import harris
import match 
import geo_veri
import draw1

import initialization
import addnew
import triangulation
import ba
import rmpoints
import reconstuction

# 读取每张图 编号为234
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
inlier_matches23, F23 = geo_veri.ransac_fundamental(corners2, corners3, matches23, thresh=50.0, max_iters=500)
inlier_matches34, F34 = geo_veri.ransac_fundamental(corners3, corners4, matches34, thresh=50.0, max_iters=500)
inlier_matches24, F24 = geo_veri.ransac_fundamental(corners2, corners4, matches24, thresh=50.0, max_iters=500)

'''
# 进行画图：特征点连线
match_img_23 = draw1.draw_matches(img2, img3, corners2, corners3, inlier_matches23)
cv2.imwrite("outputs/match_23.jpg", match_img_23)

match_img_34 = draw1.draw_matches(img3, img4, corners3, corners4, inlier_matches34)
cv2.imwrite("outputs/match_34.jpg", match_img_34)

match_img_24 = draw1.draw_matches(img2, img4, corners2, corners4, inlier_matches24)
cv2.imwrite("outputs/match_24.jpg", match_img_24)

# 画图：可视化核线
epi_23 = draw1.draw_epipolar_lines(img2, img3, F23, corners2, corners3, inlier_matches23)
cv2.imwrite("outputs/epi_23.png", epi_23)

epi_34 = draw1.draw_epipolar_lines(img3, img4, F34, corners3, corners4, inlier_matches34)
cv2.imwrite("outputs/epi_34.png", epi_34)

epi_24 = draw1.draw_epipolar_lines(img2, img4, F24, corners2, corners4, inlier_matches24)
cv2.imwrite("outputs/epi_24.png", epi_24)
'''

# 初始化

# 读取相机内外参
# K是内参，可以直接用，R和t是外参，这里作为真值r保存
K2, R2r, t2r = initialization.load_cam_file("images/00000022_cam.txt")
K3, R3r, t3r = initialization.load_cam_file("images/00000023_cam.txt")
K4, R4r, t4r = initialization.load_cam_file("images/00000024_cam.txt")

# 第一波三角化：任意两张图
# 这里选择23两张图
# 得到：2和3的相机位姿P，估计的3外参R和t，23两图的点云，以及对应的角点编号
P2, P3, points3D = initialization.initialize_two_view(corners2, corners3, inlier_matches23, K2, K3, F23)

print("P2:\n", P2)
print("P3:\n", P3)
print("init 3D point:", points3D.shape[0])

# 已有数据
P_list = [P2, P3]
points3D_init = points3D
obs_init = [(i,j) for i,j in inlier_matches23]

# 新图数据
corners4 = corners4
K4 = K4
inlier_matches34 = inlier_matches34
corners_prev = corners3  # 对应前一张图的角点

# 加入新图
P4, points3D_updated, obs_updated = addnew.add_new_image_np(
    P_list, points3D_init, obs_init, corners4, inlier_matches34, K4, corners_prev
)

print("P4:\n", P4)
print("new 3D point:", points3D_updated.shape[0])

# BA优化
# 已有
# P2, P3, P4
# points3D_init
# corners2, corners3, corners4
# inlier_matches23, inlier_matches34

P_list = [P2, P3, P4]
points3D = points3D_updated.copy()  # 已经加入初步新点的点云

# 构造 obs
obs_list, points3D_full = ba.construct_obs_safe(inlier_matches23, inlier_matches34, points3D, corners2, corners3, corners4, P_list)

# 执行 BA
P_list_opt, points3D_opt = ba.bundle_adjustment(P_list, points3D_full, obs_list, [corners2, corners3, corners4], max_iters=10, lr=1e-6)

print("BA finish")
print("3D after BA:", points3D_opt.shape[0])

# 删除错误点
points3D_clean, obs_clean = rmpoints.remove_outlier_points(
    points3D_opt, obs_list, P_list_opt, [corners2, corners3, corners4], threshold=5.0
)

print("3D cleared:", points3D_clean.shape[0])

# 重建
img_list = [img2, img3, img4]  # 原始图像，用于提取颜色
K_list = [K2, K3, K4]
R_true_list = [R2r, R3r, R4r]  # 真值旋转矩阵
t_true_list = [t2r, t3r, t4r]  # 真值平移向量

reconstuction.save_reconstruction(
    points3D_clean,
    P_list_opt,
    obs_clean,
    [corners2, corners3, corners4],
    img_list,
    K_list,
    R_true_list,
    t_true_list,
    point_txt="outputs/points_final.txt",
    pose_txt="outputs/poses_final.txt",
    error_txt="outputs/errors_final.txt"
)
