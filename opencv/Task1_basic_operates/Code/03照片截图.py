import cv2

# 读取图片
img = cv2.imread('all.jpg')

# 假设手机在左上角，宽和高分别为w和h
w, h = 300, 400
phone = img[0:h, 0:w]

# 保存截取的图片
cv2.imwrite('phone.jpg', phone)