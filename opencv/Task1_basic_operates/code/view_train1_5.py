import cv2
import numpy as np
from matplotlib.pyplot import contour

# 打开摄像头（设备索引为0）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头！")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取摄像头画面！")
        break

    # 将图像转换为 LAB 颜色空间
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

    # 定义橙色的 LAB 范围
    lower_orange = np.array([14, 140, 154])  # LAB 下限 (L, A, B)
    upper_orange = np.array([255, 180, 200])  # LAB 上限 (L, A, B)

    # 创建掩膜，仅保留橙色区域
    mask = cv2.inRange(lab, lower_orange, upper_orange)

    # 使用高斯模糊去噪
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # 查找掩膜上的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 获取最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 1000:
            # 获取轮廓的外接矩形
            x, y, w, h = cv2.boundingRect(largest_contour)

            # 计算外接矩形的中心点
            center_x, center_y = x + w // 2, y + h // 2

            point_color = frame[center_y, center_x]
            point_color1 = [max(0, min(255, c)) for c in point_color]
            inverted_color = [255 - c for c in point_color1]
            contour_color = tuple(map(int, inverted_color))
            # 绘制轮廓边缘
            cv2.drawContours(frame, [largest_contour], -1, contour_color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), contour_color, 2)
            cross_size = 10
            cv2.line(frame, (center_x - cross_size, center_y), (center_x + cross_size, center_y), contour_color, 2)
            cv2.line(frame, (center_x, center_y - cross_size), (center_x, center_y + cross_size), contour_color, 2)

    # 显示处理后的帧
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == 32:
        break
cap.release()
cv2.destroyAllWindows()
