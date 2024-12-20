# 导入相关库
import cv2
import numpy as np

# 读取图片
image = cv2.imread('E:\opencv jietu\phone.jpg')

# 让我们使用新的宽度和高度图像
kuan = 640
gao = 480
down_points = (kuan,gao)
a = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
b = cv2.resize(image, down_points, interpolation= cv2.INTER_NEAREST)
c = cv2.resize(image, down_points, interpolation= cv2.INTER_AREA)
d = cv2.resize(image, down_points, interpolation= cv2.INTER_CUBIC)
e = cv2.resize(image, down_points, interpolation= cv2.INTER_LANCZOS4)
# 显示图像
cv2.imshow('a', a)
cv2.imshow('b', b)
cv2.imshow('c', c)
cv2.imshow('d', d)
cv2.imshow('e', e)
cv2.waitKey(0)


#按下任意键退出
cv2.destroyAllWindows()
cv2.imwrite("phone_resized.jpg",a);
