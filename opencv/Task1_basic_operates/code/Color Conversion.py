import cv2
import numpy as np

# 读取图像
img = cv2.imread("D:/opencv_test/all.jpg")

# 转换为灰度图并保存
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("all_gray.jpg", gray_img)

# 转换为HSV图并保存
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite("all_hsv.jpg", hsv_img)

# 转换为Lab图并保存
lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
cv2.imwrite("all_lab.jpg", lab_img)

