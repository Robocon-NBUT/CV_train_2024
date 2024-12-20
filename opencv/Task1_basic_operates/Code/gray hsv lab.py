import cv2

# 读取all.jpg图像
image = cv2.imread('all.jpg')
if image is None:
    print("无法读取all.jpg图像，请检查文件是否存在及路径是否正确")
    exit()

# 转换为灰度颜色空间
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('all_gray.jpg', gray_image)

# 转换为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imwrite('all_hsv.jpg', hsv_image)

# 转换为LAB颜色空间
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imwrite('all_lab.jpg', lab_image)

print("已成功将all.jpg转换并保存为三种不同颜色空间的图像")
