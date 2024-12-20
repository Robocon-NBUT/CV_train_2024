import cv2
import numpy as np
from matplotlib import pyplot as plt
cap = cv2.VideoCapture(0)  # 打开摄像头

while (1):
    # get a frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([1, 120, 50])
    upper_red = np.array([10, 255, 255])

    # 通过上下限提取范围内的掩模mask
    mask = cv2.inRange(hsv, lower_red, upper_red)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    for contour in contours:
    # 找到轮廓的最小外接矩形
     rect = cv2.minAreaRect(contour)
    #获取矩形的四个顶点
    box = cv2.boxPoints(rect)
    box = np.int32(box)  # 将坐标转换为整数

    # 绘制矩形
    cv2.drawContours(frame, [box], -1, (255, 0, 0), 2)
    # 获取矩形的中心点坐标
    center = (int(rect[0][0]), int(rect[0][1]))  # 从RotatedRect中提取中心点坐标

    # 绘制中心点
    cv2.circle(frame, center, 5, (255, 0, 0), -1)
    cv2.line(frame, (center[0] - 10, center[1]), (center[0] + 10, center[1]), (0, 255, 0), 2)  # 绘制水平线
    cv2.line(frame, (center[0], center[1] - 10), (center[0], center[1] + 10), (0, 255, 0), 2)  # 绘制垂直线
    cv2.imshow("Original", frame)
    cv2.imshow("Masked", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下q 就截图保存并退出

        break

cap.release()
cv2.destroyAllWindows()