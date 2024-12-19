import cv2

# 打开 all.jpg 和 phone.jpg
all_image = cv2.imread("all.jpg")
phone_image = cv2.imread("phone.jpg")

# 检查图片是否加载成功
if all_image is None:
    print("无法读取 all.jpg 文件")
    exit()
if phone_image is None:
    print("无法读取 phone.jpg 文件")
    exit()

# 获取 all.jpg 的尺寸
all_height, all_width = all_image.shape[:2]

# 调整 phone.jpg 的尺寸与 all.jpg 一致

import cv2

# 读取 all.jpg
image = cv2.imread("all.jpg")

# 检查图片是否加载成功
if image is None:
    print("无法读取 all.jpg 文件")
    exit()

# 转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("all_gray.jpg", gray_image)

# 转换为 HSV 图
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imwrite("all_hsv.jpg", hsv_image)

# 转换为 Lab 图
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
cv2.imwrite("all_lab.jpg", lab_image)

print("已保存灰度图为 all_gray.jpg")
print("已保存 HSV 图为 all_hsv.jpg")
print("已保存 Lab 图为 all_lab.jpg")
