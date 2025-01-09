import cv2

# 读取图片
img = cv2.imread('all.jpg')

# 显示图片，让用户手动框选区域
r = cv2.selectROI("Image", img)

# 提取选中的区域（手机部分）
cropped = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

# 保存截取后的图片为phone.jpg
cv2.imwrite('phone.jpg', cropped)

# 关闭显示窗口
cv2.destroyAllWindows()