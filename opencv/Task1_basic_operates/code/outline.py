import cv2
import numpy as np
cap = cv2.VideoCapture(0)  # 0表示打开默认摄像头，可根据实际情况更改编号
print("按下q退出")
while True:
    ret, frame = cap.read()  # 读取一帧图像，ret表示是否成功读取，frame是图像数据
    
    if not ret:
        break

    # 以下为对每一帧图像进行的处理操作
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图，便于后续处理
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 进行二值化处理
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓

    for contour in contours:
        # 用相反颜色绘制轮廓
        opposite_color_contour = []
        for point in contour:
            opposite_color_point = [255 - point[0][0], 255 - point[0][1]]
            opposite_color_contour.append([opposite_color_point])
        opposite_color_contour = np.array(opposite_color_contour, dtype=np.int32)
        cv2.drawContours(frame, [opposite_color_contour], -1, (0, 0, 0), 2)  # 这里以黑色作为相反颜色示例，可根据实际调整

        # 计算外接矩形及几何中心
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2

        # 用与轮廓相同颜色绘制十字（这里假设轮廓颜色为白色，实际中可根据情况获取真实颜色）
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (255, 255, 255), 2)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (255, 255, 255), 2)
        

    cv2.imshow('Processed Image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出循环
        break

cap.release()  # 释放摄像头资源
cv2.destroyAllWindows()  # 关闭所有窗口

