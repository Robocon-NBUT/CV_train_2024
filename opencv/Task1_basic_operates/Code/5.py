import cv2
import numpy as np

# 打开摄像头，0表示默认的摄像头，如果有多个摄像头可以尝试更改参数为1、2等
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 读取一帧图像
    ret, frame = cap.read()
    if not ret:
        print("未能获取到图像帧")
        break

    # 转换为灰度图像，方便后续处理（这里假设物料与背景有一定灰度差异，可根据实际情况调整）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 进行阈值处理，提取物料轮廓，这里简单使用固定阈值127，可根据实际优化阈值选取方法
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 计算外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)  # 将坐标数据转换为整数类型

        # 获取外接矩形的几何中心坐标
        center_x = int(rect[0][0])
        center_y = int(rect[0][1])

        # 绘制物料轮廓（以相反颜色）
        # 采用固定的黑色来绘制轮廓，可根据实际情况换成白色(255, 255, 255)等其他颜色尝试效果
        cv2.drawContours(frame, [contour], -1, (0, 0, 0), 2)

        # 绘制十字标记几何中心（与轮廓相同颜色）
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 0), 2)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 0), 2)

    # 显示处理后的图像
    cv2.imshow('Frame', frame)

    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
# 关闭所有窗口
cv2.destroyAllWindows()
