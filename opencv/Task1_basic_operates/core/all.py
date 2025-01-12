import cv2

# 初始化摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 显示实时视频
    cv2.imshow('Camera', frame)

    # 按 's' 键保存图片
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('all.jpg', frame)
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()