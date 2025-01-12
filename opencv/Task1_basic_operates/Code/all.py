import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)   #0指内部摄像头，1指为外部摄像头，还可搭配 cv2.CAP_DSHOW

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 逐帧捕获
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧")
        break

    # 显示画面
    cv2.imshow('Camera', frame)

    # 等待按键
    key = cv2.waitKey(1)
    if key == 27:  # ESC键的ASCII码是27
        break
    elif key == 32:  # 空格键的ASCII码是32
        # 保存图像
        cv2.imwrite('all.jpg', frame)
        print("图片已保存")

# 释放资源
cap.release()
cv2.destroyAllWindows()