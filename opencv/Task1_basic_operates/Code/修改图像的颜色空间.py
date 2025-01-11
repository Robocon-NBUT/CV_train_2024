import cv2

# 读取图像
img = cv2.imread('all.jpg')

# 将图像转换为灰度空间
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 显示原图和灰度图
cv2.imshow('Original all', img)
cv2.imshow('Gray all', gray_img)
cv2.waitKey(0)

# 保存灰度图
cv2.imwrite('all_gray.jpg', gray_img)
