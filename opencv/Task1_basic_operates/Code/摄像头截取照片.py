import cv2

# 打开电脑摄像头，0表示默认的摄像头设备，如果有多个摄像头可以尝试更改参数
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 读取一帧画面
ret, frame = cap.read()

if ret:
    # 将读取到的画面保存为名为all.jpg的图片文件
    cv2.imwrite('all.jpg', frame)
    print("图片已成功保存")
else:
    print("未能成功获取图像帧")

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()