import cv2
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 设定红色的阈值范围（示例，可根据实际情况调整） 
    lower_maybe_red = np.array([0, 100, 150])
    upper_maybe_red = np.array([20, 255, 255])

    # 创建红色物体的掩码
    mask_red = cv2.inRange(hsv, lower_maybe_red, upper_maybe_red)
    # 中值滤波
    mask_red = cv2.medianBlur(mask_red, 5)

    # 高斯滤波
    mask_red = cv2.GaussianBlur(mask_red, (3, 3), 0)

    # 提取红色物体所在的区域
    res_red = cv2.bitwise_and(frame, frame, mask=mask_red)

    # 转换为灰度图进行边缘检测
    gray_red = cv2.cvtColor(res_red, cv2.COLOR_BGR2GRAY)
    edges_red = cv2.Canny(gray_red, 50, 150)

    # 寻找轮廓
    contours_red, _ = cv2.findContours(edges_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_red:
        area = cv2.contourArea(contour)
        # 设定一个面积阈值,只有面积大于该阈值的轮廓才绘制
        if area > 50:
            # 获取轮廓所包围区域内的像素
            x, y, w, h = cv2.boundingRect(contour)
            region = frame[y:y + h, x:x + w]
            if region.size > 0:
                # 计算平均颜色（分别对每个通道求平均）
                mean_color = np.mean(region, axis=(0, 1))
                contour_color = tuple(int(c) for c in mean_color)
            else:
                continue

            # 计算反色（通过对BGR通道值分别取反来实现）
            opposite_color = tuple(255 - int(c) for c in contour_color)

            # 用反色绘制红色物体轮廓
            cv2.drawContours(frame, [contour], 0, opposite_color, 2)

            # 计算外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # 计算几何中心
            center_x = int((box[0][0] + box[2][0]) / 2)
            center_y = int((box[0][1] + box[2][1]) / 2)

            # 用反色标记几何中心
            cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), opposite_color, 2)
            cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), opposite_color, 2)

    # 显示结果
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()