import cv2
import numpy as np
# 读取原始图像
image = cv2.imread('all.jpg')

# 将图像转换为 LAB 颜色空间
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# 显示 LAB 图像
cv2.imshow('LAB Image', lab_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

