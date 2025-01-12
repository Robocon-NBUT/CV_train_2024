import cv2

# 读取all.jpg获取其尺寸
all_img = cv2.imread('all.jpg')
height_all, width_all, _ = all_img.shape

# 读取phone.jpg
phone_img = cv2.imread('phone.jpg')
height_phone, width_phone, _ = phone_img.shape

# 计算宽度和高度的拉伸比例
width_ratio = width_all / width_phone
height_ratio = height_all / height_phone

# 按照比例拉伸图片
resized_phone_img = cv2.resize(phone_img, (width_all, height_all), interpolation=cv2.INTER_LINEAR)

# 保存拉伸后的图片为phone_resized.jpg
cv2.imwrite('phone_resized.jpg', resized_phone_img)