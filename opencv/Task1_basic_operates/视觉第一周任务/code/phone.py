import cv2
import time
# 创建一个全局变量来存储鼠标按下的起始点坐标
# 初始化裁剪框的起始和结束点，以及裁剪状态
start_point = None
end_point = None
cropping = False

# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, cropping
    img = param  # 获取传递的图像

    if event == cv2.EVENT_LBUTTONDOWN:
        # 鼠标左键按下，记录起始点
        start_point = (x, y)
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        # 鼠标左键释放，记录结束点，进行裁剪
        end_point = (x, y)
        cropping = False

        # 进行裁剪操作
        cropped_img = img[min(start_point[1], end_point[1]):max(start_point[1], end_point[1]),
                          min(start_point[0], end_point[0]):max(start_point[0], end_point[0])]

        # 保存裁剪后的图像
        cv2.imwrite('phone.jpg', cropped_img)
        print("已成功截取照片内容并保存为phone.jpg！")

# 读取原图
image = cv2.imread('all.jpg')

# 显示图像并设置鼠标回调函数
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", mouse_callback, image)

# 等待直到按下任意键退出
cv2.waitKey(0)

# 关闭所有窗口
cv2.destroyAllWindows()
