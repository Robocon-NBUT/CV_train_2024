import cv2

# 打开电脑摄像头（设备索引0为默认摄像头）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头！")
    exit()

print("按空格键拍照，按 ESC 键退出程序。")

while True:
    # 从摄像头捕获一帧图像
    ret, frame = cap.read()
    if not ret:
        print("无法获取摄像头画面！")
        break

    # 显示摄像头画面
    cv2.imshow('Camera', frame)

    # 按键监听
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 按下 ESC 键退出
        print("退出程序。")
        break
    elif key == 32:  # 按下空格键拍照
        # 压缩图像为 10% 大小
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # 保存压缩后的图像
        save_path = 'all.jpg'
        cv2.imwrite(save_path,frame)
        print(f"图片已保存为 {save_path}")

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
