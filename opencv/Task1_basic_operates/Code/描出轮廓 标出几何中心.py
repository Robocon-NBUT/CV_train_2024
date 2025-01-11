import cv2 as cv
import numpy as np

# 定义颜色范围配置字典，键为颜色名称，值为对应的HSV范围（两个元组分别表示下限和上限）
color_ranges = {
    "red": (
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255]))
    ),
    "blue": (
        (np.array([100, 120, 70]), np.array([130, 255, 255])),
    )
}

def detect_color_objects(frame, color_name):
    """
    在给定帧图像中检测指定颜色的物体，并绘制轮廓、矩形框和几何中心标记
    """
    # 将BGR帧转换为HSV颜色空间
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # 获取指定颜色的HSV范围
    if color_name in color_ranges:
        if isinstance(color_ranges[color_name][0], tuple):  # 处理像红色那样有两个范围的情况
            lower_red1, upper_red1 = color_ranges[color_name][0]
            lower_red2, upper_red2 = color_ranges[color_name][1]
            mask1 = cv.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv.inRange(hsv, lower_red2, upper_red2)
            mask = cv.add(mask1, mask2)
        else:  # 处理只有一个范围的颜色情况
            lower, upper = color_ranges[color_name][0]
            mask = cv.inRange(hsv, lower, upper)
    else:
        print(f"未定义 {color_name} 的颜色范围，请检查配置。")
        return frame

    # 对掩膜进行形态学操作，去除噪点（可选）
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # 找到指定颜色区域的轮廓
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 在原图上绘制轮廓以及对应的矩形框，并标记几何中心
    for contour in contours:
        if cv.contourArea(contour) > 500:  # 过滤掉较小的轮廓
            # 绘制轮廓
            cv.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            # 获取包围轮廓的矩形框坐标
            x, y, w, h = cv.boundingRect(contour)
            # 绘制矩形框
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 计算矩形的几何中心坐标
            center_x = x + w // 2
            center_y = y + h // 2
            # 以十字标记几何中心，线长设为10（可根据实际情况调整）
            cv.line(frame, (center_x - 5, center_y), (center_x + 5, center_y), (0, 255, 0), 2)
            cv.line(frame, (center_x, center_y - 5), (center_x, center_y + 5), (0, 255, 0), 2)

    return frame


# 打开摄像头
cap = cv.VideoCapture(0)

while True:
    # 读取摄像头的一帧
    ret, frame = cap.read()
    if not ret:
        print("无法从摄像头获取帧")
        break

    # 检测红色物体（这里可通过修改参数检测其他颜色物体）
    frame = detect_color_objects(frame, "red")

    # 显示原始帧（带有物体轮廓、矩形框和几何中心标记）
    cv.imshow('Original Frame', frame)

    k = cv.waitKey(1)
    if k == 27:
        # 通过esc键退出摄像
        cv.destroyAllWindows()
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv.destroyAllWindows()