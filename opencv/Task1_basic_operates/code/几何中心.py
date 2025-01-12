import cv2
import numpy as np

# 读取图像
image = cv2.imread('wujian.jpg')

# 将图像转换为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义橙色的HSV范围（可根据实际情况调整）
lower_orange = np.array([0, 100, 100])
upper_orange = np.array([20, 255, 255])

# 创建橙色的掩码
orange_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

# 查找橙色物体的轮廓
contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历轮廓
for contour in contours:
    # 计算外接矩形
    x, y, w, h = cv2.boundingRect(contour)
    # 计算几何中心
    center_x = x + w // 2
    center_y = y + h // 2
 # 绘制外接矩形（用与轮廓相同的颜色，这里假设为蓝色，可根据实际颜色修改）
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # 绘制几何中心（用与轮廓相同的颜色十字，这里假设为蓝色，可根据实际颜色修改）
    cv2.line(image, (center_x - 10, center_y), (center_x + 10, center_y), (255, 0, 0), 2)
    cv2.line(image, (center_x, center_y - 10), (center_x, center_y + 10), (255, 0, 0), 2)
# 显示结果
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
