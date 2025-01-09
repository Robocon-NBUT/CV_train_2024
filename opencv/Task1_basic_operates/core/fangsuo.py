import cv2

# 读取图片
phone_img = cv2.imread('phone.jpg')
all_img = cv2.imread('all.jpg')

# 获取原图尺寸
height, width = all_img.shape[:2]

# 调整尺寸
resized_phone_img = cv2.resize(phone_img, (width, height))

# 保存调整后的图片
cv2.imwrite('phone_resized.jpg', resized_phone_img)