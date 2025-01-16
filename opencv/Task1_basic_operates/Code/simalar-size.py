import cv2

# 读取all.jpg图像，获取其尺寸
all_image = cv2.imread('all.jpg')
if all_image is None:
    print("无法读取all.jpg文件，请检查文件是否存在及路径是否正确")
    exit()
height_all, width_all, _ = all_image.shape

# 读取phone.jpg图像
phone_image = cv2.imread('phone.jpg')
if phone_image is None:
    print("无法读取phone.jpg文件，请检查文件是否存在及路径是否正确")
    exit()

# 获取phone.jpg图像当前的尺寸
height_phone, width_phone, _ = phone_image.shape

# 计算缩放比例
scale_x = width_all / width_phone
scale_y = height_all / height_phone

# 根据缩放比例对phone.jpg图像进行缩放
resized_phone_image = cv2.resize(phone_image, (0, 0), fx=scale_x, fy=scale_y)

# 保存缩放后的图像为phone_resized.jpg
cv2.imwrite('phone_resized.jpg', resized_phone_image)
print("phone.jpg已成功缩放并保存为phone_resized.jpg")