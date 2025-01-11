import cv2

# 读取图片
img = cv2.imread('all.jpg')  # 替换为实际的图片文件名及路径

# 获取图片的高度、宽度和通道数（这里通道数一般是3表示RGB三个通道）
height, width, _ = img.shape

# 定义要截取区域的左上角坐标以及截取区域的宽度和高度
x = 460 # 示例左上角x坐标，可按需修改
y = 400  # 示例左上角y坐标，可按需修改
w = 460 # 截取区域宽度，可按需修改
h = 200 # 截取区域高度，可按需修改

# 截取图片
cropped_img = img[y:y + h, x:x + w]

# 显示截取后的图片（可选，用于查看截取效果，按任意键关闭显示窗口）
cv2.imshow('Cropped Image', cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存截取后的图片
cv2.imwrite('phone.jpg', cropped_img)  # 可按需更改保存的文件名及路径