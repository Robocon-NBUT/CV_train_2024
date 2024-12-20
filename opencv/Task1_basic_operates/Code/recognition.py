import cv2
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头的一帧画面
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像从BGR转换到HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义红色在HSV中的范围
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # 合并两个掩码
    mask = mask1 + mask2

    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓和中心点
    for contour in contours:
        # 计算轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2
        # 绘制轮廓（青色）
        cv2.drawContours(frame, [contour], -1, (255, 255, 0), 2)
        # 绘制中心点（青色）
        cv2.drawMarker(frame, (center_x, center_y), (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

    # 显示结果
    cv2.imshow('Result', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
