import cv2
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0代表默认的摄像头

# 检查摄像头是否打开成功
if not cap.isOpened():
    print("无法访问摄像头")
    exit()

# 设置摄像头分辨率（可选）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # 捕获每一帧图像
    ret, frame = cap.read()

    # 如果无法捕获帧，退出
    if not ret:
        print("无法获取视频帧")
        break

    # 转换为 HSV 颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义红色的范围
    lower_red1 = np.array([0, 120, 70])    # 红色的下限 (低色调)
    upper_red1 = np.array([10, 255, 255])  # 红色的上限 (高色调)
    lower_red2 = np.array([170, 120, 70])  # 红色的下限 (高色调)
    upper_red2 = np.array([180, 255, 255]) # 红色的上限 (低色调)

    # 创建两个掩码，用于提取红色
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)  # 合并两个掩码

    # 使用掩码提取红色区域
    red_area = cv2.bitwise_and(frame, frame, mask=mask)

    # 转换为灰度图像进行轮廓检测
    gray = cv2.cvtColor(red_area, cv2.COLOR_BGR2GRAY)

    # 使用高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 查找轮廓
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个副本，用于绘制结果
    result = frame.copy()

    for contour in contours:
        # 如果轮廓的面积太小，则跳过
        if cv2.contourArea(contour) < 500:  # 可以根据实际情况调整最小面积
            continue

        # 计算外接矩形
        x, y, w, h = cv2.boundingRect(contour)

        # 计算外接矩形的几何中心
        center = (x + w // 2, y + h // 2)

        # 绘制轮廓的相反颜色（假设背景是黑色，轮廓使用白色）
        cv2.drawContours(result, [contour], -1, (255, 255, 255), 2)

        # 绘制十字标记外接矩形的几何中心
        cv2.drawMarker(result, center, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # 显示实时处理结果
    cv2.imshow('Real-time Object Detection (Red)', result)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
