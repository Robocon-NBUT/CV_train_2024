import cv2
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # 捕获一帧图像
    ret, frame = cap.read()
    if not ret:
        break  # 如果读取失败，退出

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用高斯模糊来平滑图像，减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, 100, 200)

    # 使用形态学操作 (闭运算) 填补物体轮廓中的空洞
    kernel = np.ones((5, 5), np.uint8)  # 结构元素大小
    dilated = cv2.dilate(edges, kernel, iterations=1)  # 膨胀操作
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)  # 闭运算填补空洞

    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 如果轮廓面积过小，跳过
        if cv2.contourArea(contour) < 100:
            continue

        # 获取外接矩形的坐标和尺寸
        x, y, w, h = cv2.boundingRect(contour)

        # 计算外接矩形的几何中心
        center_x, center_y = x + w // 2, y + h // 2

        # 设置颜色为白色 (BGR: 255, 255, 255)
        white_color = (255, 255, 255)

        # 绘制轮廓，使用白色
        cv2.drawContours(frame, [contour], -1, white_color, 2)

        # 在几何中心绘制十字标记，使用白色
        cv2.drawMarker(frame, (center_x, center_y), white_color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

    # 显示实时视频
    cv2.imshow('Webcam', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭所有OpenCV窗口
cap.release()
cv2.destroyAllWindows()

