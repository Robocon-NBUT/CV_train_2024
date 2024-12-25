import cv2 as cv

# 读取图片
image = cv.imread('all.jpg')
# 将 BGR 图片转换为 HSV 图像
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
# 使用cv2.imwrite保存HSV图像，第一个参数为保存的文件名，第二个参数为要保存的图像（即hsv_image）
cv.imwrite("all_hsv.jpg", hsv_image)
# 显示原图和 HSV 图像
cv.imshow('Original Image', image)
cv.imshow('HSV Image', hsv_image)


cv.waitKey(0)
cv.destroyAllWindows()