import cv2
import numpy as np

# 读取图片
img = cv2.imread("D:/opencv_test/all.jpg")

# 截取手机部分（这里假设坐标，需根据实际修改）
x1, y1, x2, y2 = 311, 92, 478, 440
phone_part = img[y1:y2, x1:x2]

# 正常保存图片
cv2.imwrite("phone.jpg", phone_part)


                                       