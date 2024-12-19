import cv2

# 打开默认摄像头（参数 0 表示默认摄像头）
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("无法打开摄像头")
    exit()

print("请对准手机，按 's' 拍摄并保存图片，按 'q' 退出")

while True:
    # 读取摄像头帧
    ret, frame = camera.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    # 显示摄像头画面
    cv2.imshow('Camera', frame)

    # 按键事件
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # 按 's' 保存图片
        cv2.imwrite('all.jpg', frame)
        print("图片已保存为 all.jpg")
    elif key == ord('q'):  # 按 'q' 退出
        break

# 释放摄像头资源
camera.release()
cv2.destroyAllWindows()
