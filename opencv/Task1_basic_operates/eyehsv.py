import cv2
import numpy as np


# 读取原始图像
image = cv2.imread('all.jpg')

# 将图像转换为 HSV 颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 显示 HSV 图像
cv2.imshow('HSV Image', hsv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


