import cv2
import numpy as np

def draw_matches(img1, img2, corners1, corners2, matches):
    """
    可视化匹配点及连线
    """
    # 拼接图像
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1+w2] = img2

    for i, j in matches:
        x1, y1 = corners1[i]
        x2, y2 = corners2[j]
        x2 += w1  # 右图坐标平移

        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(canvas, (int(x1), int(y1)), 3, color, -1)
        cv2.circle(canvas, (int(x2), int(y2)), 3, color, -1)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

    return canvas   


# 可视化核线
def draw_epipolar_lines(img1, img2, F, pts1, pts2, matches, max_lines=20, color=(0,255,0)):
    """
    img1, img2: BGR images
    F: 3x3 fundamental matrix
    pts1, pts2: Nx2 float coordinates
    matches: list of (i,j)
    max_lines: only draw first N lines
    """
    
    # 拼接图像
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas_h = max(h1, h2)
    canvas = np.zeros((canvas_h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    # 右侧图像的 x 偏移
    offset = np.array([w1, 0])

    # 画线函数：给定 l=[a b c]，画 ax+by+c=0
    def draw_line(img, line, color):
        a, b, c = line
        x0, y0 = 0, int(-c / b) if b != 0 else 0
        x1, y1 = img.shape[1], int(-(c + a * img.shape[1]) / b) if b != 0 else 0
        cv2.line(img, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

    # 限制数量
    total = min(max_lines, len(matches))

    # 双向核线绘制
    for k in range(total):
        i, j = matches[k]

        p1 = np.array([pts1[i][0], pts1[i][1], 1.0])
        p2 = np.array([pts2[j][0], pts2[j][1], 1.0])

        # 图1 ← 图2 的核线： l1 = F^T p2
        l1 = F.T @ p2

        # 图2 ← 图1 的核线： l2 = F p1
        l2 = F @ p1

        # 在图1画来自图2的核线
        draw_line(canvas[:, :w1], l1, (0,255,0))

        # 在图2画来自图1的核线（需要加 offset）
        line_offset = np.array([l2[0], l2[1], l2[2] - l2[0]*offset[0] - l2[1]*offset[1]])
        draw_line(canvas[:, w1:], line_offset, (0,255,0))

    return canvas
