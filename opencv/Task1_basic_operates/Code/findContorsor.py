import cv2
import numpy as np

# 打开电脑摄像头，0表示默认的摄像头设备
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 设置面积阈值，单位为像素面积，你可根据实际情况调整该值
area_threshold = 10000

while True:
    # 读取摄像头的一帧画面
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像从BGR颜色空间转换为HSV颜色空间，方便根据颜色范围筛选橙色
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 调整橙色在HSV颜色空间中的范围
    lower_orange = np.array([0, 80, 80])
    upper_orange = np.array([30, 255, 255])

    # 根据橙色的HSV范围创建掩膜，提取出橙色部分的图像
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # 对提取出的橙色部分图像进行直方图均衡化，增强对比度（先将单通道图像转换为合适格式）
    mask = cv2.equalizeHist(mask)

    # 使用阈值处理，将图像二值化，这里采用自适应阈值，可根据实际情况调整参数
    thresh = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 创建结构元素，这里使用矩形结构元素，大小可根据实际调整，例如(3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 先进行腐蚀操作，去除小噪点等，迭代次数设为1（可调整）
    thresh = cv2.erode(thresh, kernel, iterations=1)
    # 再进行膨胀操作，恢复一些被腐蚀掉的轮廓部分（如果有的话），迭代次数设为1（可调整）
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 计算轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)

        # 计算轮廓的面积
        contour_area = cv2.contourArea(contour)

        # 判断轮廓面积是否大于设定的阈值，如果大于则进行后续操作
        if contour_area > area_threshold:
            # 获取外接矩形的几何中心坐标
            center_x = x + w // 2
            center_y = y + h // 2

            # 获取轮廓区域内某一点的颜色（这里取外接矩形左上角的点作为代表，可根据实际优化）
            color = frame[y, x].tolist()
            color = tuple(int(c) for c in color)

            # 反转颜色，用于标记轮廓
            reversed_color = tuple(255 - c for c in color)

            # 绘制物料轮廓，用反转后的颜色
            cv2.drawContours(frame, [contour], -1, reversed_color, 2)

            # 用与轮廓相同的颜色绘制十字，标记外接矩形的几何中心
            cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), color, 2)
            cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), color, 2)

            # 添加绘制矩形框的代码，使用绿色（BGR格式为(0, 255, 0)）框住物料，框线宽度为2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示处理后的图像
    cv2.imshow('Processed Image', frame)

    # 等待按键事件，按下 'q' 键退出循环
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("即将进行截图，请保持画面稳定...")
        cv2.imwrite('a.jpg', frame)
        cv2.imwrite(save_path, frame)
        print("截图已成功保存")

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()