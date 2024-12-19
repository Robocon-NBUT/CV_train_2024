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
resized_phone = cv2.resize(phone_image, (all_width, all_height))

# 保存为 phone_resized.jpg
cv2.imwrite("phone_resized.jpg", resized_phone)
print("已保存调整大小后的图片为 phone_resized.jpg")
