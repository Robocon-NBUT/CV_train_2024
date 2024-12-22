import cv2

# 回调函数，用于鼠标操作
def crop_image(event, x, y, flags, param):
    global cropping, start_x, start_y, end_x, end_y, image, clone

    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键按下，记录起始坐标
        cropping = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and cropping:  # 鼠标移动，实时显示选区
        temp_image = clone.copy()
        cv2.rectangle(temp_image, (start_x, start_y), (x, y), (0, 255, 0), 2)
        cv2.imshow("all.jpg", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键抬起，记录结束坐标
        cropping = False
        end_x, end_y = x, y

        # 画出最终选区
        cv2.rectangle(clone, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.imshow("all.jpg", clone)

        # 裁剪选定区域并保存
        crop = image[start_y:end_y, start_x:end_x]
        save_path = 'phone.jpg'
        cv2.imwrite(save_path, crop)
        print(f"裁剪区域已保存为 {save_path}")

# 初始化变量
cropping = False
start_x = start_y = end_x = end_y = 0

# 打开 all.jpg 图像
image = cv2.imread('all.jpg')
if image is None:
    print("无法打开 all.jpg，请检查文件路径。")
    exit()

clone = image.copy()  # 克隆图像用于绘制

# 显示图像并绑定鼠标事件
cv2.namedWindow("all.jpg")
cv2.setMouseCallback("all.jpg", crop_image)
cv2.imshow("all.jpg", image)

print("请用鼠标框选手机区域。左键按下开始框选，释放左键完成裁剪。")
cv2.waitKey(0)
cv2.destroyAllWindows()
