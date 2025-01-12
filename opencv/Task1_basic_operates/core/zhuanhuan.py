import cv2

# 读取图片
img = cv2.imread('all.jpg')

# 转换为灰度
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('all_gray.jpg', gray_img)

# 转换为 HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite('all_hsv.jpg', hsv_img)

# 转换为 LAB
lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imwrite('all_lab.jpg', lab_img)