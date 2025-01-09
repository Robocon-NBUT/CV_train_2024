import cv2

# 定义红色在LAB颜色空间中的大致范围
lower_red_a = 150
upper_red_a = 255
lower_red_b = 150
upper_red_b = 255

# 打开摄像头，0表示默认摄像头，可根据实际情况调整
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头的一帧画面
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像从BGR颜色空间转换为LAB颜色空间
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # 分离LAB通道
    l_channel, a_channel, b_channel = cv2.split(lab_frame)

    # 创建红色的掩膜，基于LAB空间的a和b通道范围筛选红色区域
    mask_a = cv2.inRange(a_channel, lower_red_a, upper_red_a)
    mask_b = cv2.inRange(b_channel, lower_red_b, upper_red_b)
    mask = cv2.bitwise_and(mask_a, mask_b)

    # 对掩膜进行形态学操作（可选，可去除一些小噪声等）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 查找红色区域的轮廓
    contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 计算外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(int)

        # 获取外接矩形的几何中心坐标
        center_x = int(rect[0][0])
        center_y = int(rect[0][1])

        # 用蓝色绘制红色小积木的轮廓，在BGR颜色空间中蓝色为(255, 0, 0)
        cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

        # 用蓝色绘制十字，标记外接矩形的几何中心
        cv2.line(frame, (center_x - 5, center_y), (center_x + 5, center_y), (255, 0, 0), 2)
        cv2.line(frame, (center_x, center_y - 5), (center_x, center_y + 5), (255, 0, 0), 2)

    # 显示处理后的画面
    cv2.imshow('Detected Red Blocks', frame)

    # 按下ESC键（ASCII码为27）退出循环
    key = cv2.waitKey(1)
    if key == 27:
        break

# 释放摄像头资源
cap.release()
# 关闭所有窗口
cv2.destroyAllWindows()