import cv2

# 读取图像文件，替换为你实际的图像文件路径
image_path = "all.jpg"
image = cv2.imread(image_path)
if image is None:
    print("无法读取图像文件，请检查文件路径是否正确")
    exit()

# 确定手机部分在图像中的坐标范围（需根据实际情况调整）
x = 5
y = 105
w = 616
h = 290

# 截取手机部分图像
phone_image = image[y:y + h, x:x + w]

# 显示截取后的手机图像
cv2.imshow('Phone Image', phone_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存截取后的手机图像（可选步骤，替换为你想要保存的文件名和路径）
save_path = "phone.jpg"
cv2.imwrite(save_path, phone_image)
print(f"截取的手机图像已成功保存至 {save_path}")