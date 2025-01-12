import cv2
import numpy as np



def fan_se(color):
    #返回给定颜色的相反颜色
    return (255 - color[0], 255 - color[1], 255 - color[2])

def shi_zi(img, center, color, size=10, thickness=2):
    #在指定位置绘制十字标记
    x, y = center
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)

def shibie():
    # 打开摄像头（默认摄像头ID为0）
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 定义橙色的HSV范围
    # 这些值可能需要根据实际情况调整
    lower_orange = np.array([1, 120, 50])
    upper_orange = np.array([10, 255, 255])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取摄像头画面")
            break

        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 创建橙色的掩模
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # 使用形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        # 查找轮廓
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 假设目标物体是最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)

            # 仅在轮廓面积足够大时处理
            if cv2.contourArea(largest_contour) > 500:
                # 绘制轮廓的外接矩形
                x, y, w, h = cv2.boundingRect(largest_contour)
                center = (x + w // 2, y + h // 2)

                # 获取轮廓的颜色（这里取轮廓区域的平均颜色）
                mask_contour = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.drawContours(mask_contour, [largest_contour], -1, 255, -1)
                mean_val = cv2.mean(frame, mask=mask_contour)[:3]
                mean_color = tuple(map(int, mean_val))
                opposite_color = fan_se(mean_color)

                # 绘制轮廓
                cv2.drawContours(frame, [largest_contour], -1, opposite_color, 2)

                # 绘制几何中心的十字
                shi_zi(frame, center, mean_color)

                # 绘制外接矩形
                cv2.rectangle(frame, (x, y), (x + w, y + h), opposite_color, 2)


        # 显示结果
        cv2.imshow('shibie', frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()



shibie()
