import cv2


# 鼠标回调函数，用于在鼠标移动时显示坐标
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"鼠标位置: ({x}, {y})")


image = cv2.imread('all.jpg')
# 创建图像显示窗口
cv2.namedWindow('Image')
# 设置鼠标回调函数，关联到图像显示窗口
cv2.setMouseCallback('Image', mouse_callback)
cv2.imshow('Image', image)

# 等待按键操作，这里按任意键继续下一步
cv2.waitKey(0)


# 比如手机区域左上角坐标为(x1, y1)，右下角坐标为(x2, y2)
x1, y1 = 150, 100 
x2, y2 = 560, 290 

# 截取手机部分图像
phone_image = image[y1:y2, x1:x2]

# 将截取到的手机图像保存为phone.jpg
cv2.imwrite('phone.jpg', phone_image)

print("手机部分图像已成功保存为phone.jpg")

# 关闭所有窗口
cv2.destroyAllWindows()
