import cv2

# 打开电脑摄像头，0 表示默认的摄像头设备（如果有多个摄像头，可尝试更换为1、2等数字来选择不同摄像头）
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头，请检查设备是否正常连接。")
    exit()

# 给用户留出准备时间，例如5秒钟，可以根据实际情况调整等待时长
print("请将手机摆放至合适位置，准备拍摄，5秒后将自动拍摄。")
for i in range(5):
    print(5 - i)
    cv2.waitKey(1000)

# 从摄像头读取一帧画面
ret, frame = cap.read()

if ret:
    # 定义保存图片的文件名
    save_filename = "all.jpg"
    # 保存图片到当前目录下（你也可以指定具体的绝对路径来保存到其他位置）
    cv2.imwrite(save_filename, frame)
    print(f"图片已成功保存为 {save_filename}")
else:
    print("未能成功捕获图像，请检查摄像头或重试。")

# 释放摄像头资源
cap.release()







def on_mouse(event, x, y, flags, param):
    global refPt, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        # 绘制矩形框选区域（可选，方便可视化看到框选的区域）
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


# 读取图片
image = cv2.imread('all.jpg')
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_mouse)

refPt = []
cropping = False

while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        image = clone.copy()
    elif key == ord("c") and len(refPt) == 2:
        # 获取真正的裁剪坐标
        x, y = min(refPt[0][0], refPt[1][0]), min(refPt[0][1], refPt[1][1])
        w = abs(refPt[0][0] - refPt[1][0])
        h = abs(refPt[0][1] - refPt[1][1])
        cropped_image = image[y:y + h, x:x + w]
        cv2.imwrite('phone.jpg', cropped_image)

        print(f"图片已成功保存为phone.jpg ")
        break
    elif key == 27:
        break

all_img = cv2.imread('all.jpg')
height, width, _ = all_img.shape

# 读取phone.jpg
phone_img = cv2.imread('phone.jpg')

# 缩放phone.jpg至与all.jpg相同尺寸
resized_phone_img = cv2.resize(phone_img, (width, height))

# 保存缩放后的图片为phone_resized.jpg
cv2.imwrite('phone_resized.jpg', resized_phone_img)
print(f"重置大小图片已成功保存为phone_resized.jpg")


img = cv2.imread('all.jpg')

# 转换为灰度颜色空间
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('all_gray.jpg', gray_img)

# 转换为HSV颜色空间
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite('all_hsv.jpg', hsv_img)

# 转换为LAB颜色空间
lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imwrite('all_lab.jpg', lab_img)
print(f"更改颜色图片已成功保存为all_gray.jpg,all_hsv.jpg,all_lab.jpg")

cv2.destroyAllWindows()
    