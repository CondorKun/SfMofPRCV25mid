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