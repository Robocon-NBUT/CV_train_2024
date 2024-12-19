import cv2

# 定义全局变量
ref_point = []  # 用于存储矩形区域的顶点
cropping = False  # 是否正在进行裁剪


def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping

    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键按下
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键释放
        ref_point.append((x, y))
        cropping = False

        # 画出矩形区域
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("Image", image)


# 读取图片
image = cv2.imread("all.jpg")
if image is None:
    print("无法读取 all.jpg 文件")
    exit()

clone = image.copy()
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", click_and_crop)

print("请用鼠标选择手机区域，按 'c' 确认裁剪，按 'q' 退出")

while True:
    cv2.imshow("Image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):  # 按 'r' 重置图片
        image = clone.copy()
    elif key == ord("c"):  # 按 'c' 确认裁剪
        if len(ref_point) == 2:
            # 截取选定区域
            phone = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            cv2.imwrite("phone.jpg", phone)
            print("已保存裁剪后的图片为 phone.jpg")
        break
    elif key == ord("q"):  # 按 'q' 退出
        break

cv2.destroyAllWindows()
