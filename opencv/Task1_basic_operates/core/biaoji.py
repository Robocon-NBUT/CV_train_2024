import cv2
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义橙红色的HSV范围
    lower_orange_red = np.array([5, 100, 100])  # 下限
    upper_orange_red = np.array([10, 255, 255])  # 上限

    # 创建掩码
    mask = cv2.inRange(hsv, lower_orange_red, upper_orange_red)

    # 中值滤波
    mask = cv2.medianBlur(mask, 3)

    # 高斯滤波
    blurred = cv2.GaussianBlur(mask, (3, 3), 0)

    # Canny边缘检测
    edges = cv2.Canny(blurred, 100, 200)

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历轮廓
    for contour in contours:
        # 过滤小面积的轮廓
        area = cv2.contourArea(contour)
        if area > 100:  # 只处理面积大于100的轮廓
            # 用白色绘制轮廓
            cv2.drawContours(frame, [contour], -1, (255, 255, 255), 2)

            # 计算外接矩形
            x, y, w, h = cv2.boundingRect(contour)

            # 计算几何中心
            center_x = x + w // 2
            center_y = y + h // 2

            # 用白色绘制十字标记
            length = 10  # 十字线的长度
            cv2.line(frame, (center_x - length, center_y), (center_x + length, center_y), (255, 255, 255), 2)  # 水平线
            cv2.line(frame, (center_x, center_y - length), (center_x, center_y + length), (255, 255, 255), 2)  # 垂直线

    # 显示结果
    cv2.imshow('Camera', frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()