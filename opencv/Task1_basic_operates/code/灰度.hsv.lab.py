import cv2

# 读取图片
img = cv2.imread('all.jpg')

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('all_gray.jpg', gray)

# 转换为HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite('all_hsv.jpg', hsv)

# 转换为LAB
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imwrite('all_lab.jpg', lab)