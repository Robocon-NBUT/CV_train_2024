import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
        print("无法打开摄像头")

while True:
    ret, frame = cap.read()  # 读取帧
    if not ret:
        print("无法获取图像")
        break

    cv2.imshow('Camera Resolution', frame)  # 显示图像

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # 按下 'c' 键保存当前帧
        cv2.imwrite('all.jpg', frame)
        print("已保存照片")
    elif key == ord('q'):
        # 按下 'q' 键退出程序
        print("退出程序")
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
