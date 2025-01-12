import cv2

# 读取图像
image = cv2.imread('all.jpg')

# 检查图像是否成功加载
if image is None:
    print("图像加载失败，请检查文件路径")
else:
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('all_gray.jpg', gray_image)

    # 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite('all_hsv.jpg', hsv_image)

    # 转换为LAB颜色空间
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    cv2.imwrite('all_lab.jpg', lab_image)

    print("图像保存成功")
