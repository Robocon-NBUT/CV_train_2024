import cv2

# 读取图像
img = cv2.imread('phone.jpg')

# 调整图像大小
resized_img = cv2.resize(img, (1280, 720))

# 显示原图和调整后的图像
cv2.imshow('Original all', img)
cv2.imshow('Resized all', resized_img)
cv2.waitKey(0)

# 保存调整后的图像
cv2.imwrite('phone_resized.jpg', resized_img)
