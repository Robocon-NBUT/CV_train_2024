import cv2

# 读取图片
img = cv2.imread('all.jpg')

# 假设手机部分在图片的左上角
x, y, w, h = 187, 286, 295, 113
phone_img = img[y:y+h, x:x+w]

# 保存截取的部分
cv2.imwrite('phone.jpg', phone_img)