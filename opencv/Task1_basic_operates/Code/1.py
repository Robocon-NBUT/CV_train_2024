import cv2

# 打开摄像头，0表示默认的摄像头，如果有多个摄像头可以尝试更改参数为1、2等
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 读取一帧图像
ret, frame = cap.read()

# 释放摄像头资源
cap.release()

# 如果成功读取到了图像帧
if ret:
    # 将图像保存为all.jpg，这里可以根据实际需求调整保存的路径等
    cv2.imwrite('all.jpg', frame)
    print("图像已成功保存")
else:
    print("未能获取到图像帧")
