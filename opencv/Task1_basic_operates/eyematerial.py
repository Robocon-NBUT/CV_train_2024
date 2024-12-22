import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([30, 255, 255])
    orange_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

    kernel = np.ones((3, 3), np.uint8)
    orange_mask_eroded = cv2.erode(orange_mask, kernel, iterations = 1)
    orange_mask_dilated = cv2.dilate(orange_mask_eroded, kernel, iterations = 1)

    contours, hierarchy = cv2.findContours(orange_mask_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2
        # 计算轮廓颜色（这里以简单的反色为例）
        contour_color = [255 - 0, 255 - 255, 255 - 0]
        # 计算外接矩形颜色（反色）
        rect_color = [255 - contour_color[0], 255 - contour_color[1], 255 - contour_color[2]]
        # 绘制外接矩形
        cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)
        # 用计算出的颜色绘制十字标记中心
        cv2.line(frame, (center_x - 5, center_y), (center_x + 5, center_y), rect_color, 2)
        cv2.line(frame, (center_x, center_y - 5), (center_x, center_y + 5), rect_color, 2)
        # 绘制轮廓
        cv2.drawContours(frame, [contour], -1, contour_color, 3)
    cv2.imshow('Contours of Orange Objects with Center Mark and Bounding Rect', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


