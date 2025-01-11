import cv2

# 读取图像
image = cv2.imread('all.jpg')

# 将图像从BGR色彩空间转换为HSV色彩空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 可以选择显示转换后的HSV图像（如果需要查看效果）
cv2.imshow('all_hsv', hsv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 如果你想保存转换后的HSV图像，可以使用以下代码（保存为名为'hsv_result.jpg'的图像）
cv2.imwrite('all_hsv.jpg', hsv_image)