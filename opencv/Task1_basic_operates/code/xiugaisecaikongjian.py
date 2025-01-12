import cv2

# 读取图像
src_image = cv2.imread('E:\opencv xiugaichicun\phone_resized.jpg')

# 将BGR图像转换为灰度图像
gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)

# 将BGR图像转换为HSV图像
hsv_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)

# 将BGR图像转换为LAB图像
lab_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2LAB)

# 显示图像
cv2.imshow('Original', src_image)
#cv2.imshow('Gray', gray_image)
cv2.imshow('HSV', hsv_image)
#cv2.imshow('LAB', lab_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("all_gray.jpg",gray_image);
cv2.imwrite("all_hsv.jpg",hsv_image);
cv2.imwrite("all_lab.jpg",lab_image);
