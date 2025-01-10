import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

# 读取一帧
ret, frame = cap.read()

# 保存图片
cv2.imwrite('all.jpg', frame)

# 释放摄像头
cap.release()