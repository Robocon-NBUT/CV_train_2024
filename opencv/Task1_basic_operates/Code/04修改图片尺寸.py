import cv2

# 读取phone.jpg
phone = cv2.imread('phone.jpg')

# 读取all.jpg获取尺寸
all_img = cv2.imread('all.jpg')
height, width, channels = all_img.shape

# 缩放phone.jpg
phone_resized = cv2.resize(phone, (width, height))

# 保存缩放后的图片
cv2.imwrite('phone_resized.jpg', phone_resized)