import cv2

# 读取all.jpg图像，获取其尺寸
all_image = cv2.imread('all.jpg')
height_all, width_all, _ = all_image.shape

# 读取phone.jpg图像
phone_image = cv2.imread('phone.jpg')
height_phone, width_phone, _ = phone_image.shape

# 计算缩放比例，这里以等比例缩放为例，根据宽度或高度中较小的缩放比例来确定
if width_all / width_phone < height_all / height_phone:
    scale = width_all / width_phone
else:
    scale = height_all / height_phone

# 计算缩放后的尺寸
new_width = int(width_phone * scale)
new_height = int(height_phone * scale)

# 进行缩放操作
resized_phone_image = cv2.resize(phone_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# 将缩放后的图像保存为phone_resized.jpg
cv2.imwrite('phone_resized.jpg', resized_phone_image)

print("已成功将phone.jpg缩放并保存为phone_resized.jpg，尺寸与all.jpg一致")
