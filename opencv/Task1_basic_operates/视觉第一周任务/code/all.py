import cv2
import time
# 创建一个VideoCapture对象，参数0表示打开默认摄像
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 用于标记是否进行拍照，初始化为False
take_photo = False
cropped = False  # 新增变量，用于标记是否已经完成裁剪

while True:
    # 逐帧读取视频
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧")
        break
    # 显示帧
    cv2.imshow('摄像头画面', frame)

    # 这里通过检测按键来决定是否拍照，按下's'键（你也可以改成其他按键）进行拍照
    key = cv2.waitKey(1)
    if key == ord('s'):
        take_photo = True

    # 如果take_photo变为True，说明要拍照，执行拍照操作并保存图片
    if take_photo:
        cv2.imwrite('all.jpg', frame)
        print("照片已成功拍摄并保存！")

        # 读取刚刚拍摄保存的照片
        img = cv2.imread('all.jpg')
        cv2.namedWindow('Select Area')
        cv2.setMouseCallback('Select Area', mouse_callback, img)
        cv2.imshow('Select Area', img)
        while True:
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
        cropped = True  # 标记已经完成裁剪
        break
    # 等待1毫秒，按下'q'键退出预览画面（不拍照直接退出）
    elif key == ord('q'):
        break