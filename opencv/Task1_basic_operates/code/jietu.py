import cv2

# 读取图像
img = cv2.imread('E:/opencv take photo/all.jpg')  # 使用正斜杠

# 确定手机在图像中的位置
# 假设手机的左上角坐标为(x1, y1)，右下角坐标为(x2, y2)
# 你需要根据实际情况调整这些值
x1, y1 = 290, 120  # 示例坐标
x2, y2 = 435, 320 # 示例坐标

# 截取手机部分
a= img[y1:y2, x1:x2]

# 保存截取的图像
cv2.imwrite('phone.jpg', a)

# 显示截取的图像
cv2.imshow('Phone Image', a)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("phone.jpg",a);