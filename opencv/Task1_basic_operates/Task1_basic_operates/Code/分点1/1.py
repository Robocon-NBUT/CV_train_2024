import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

ret, frame = cap.read()

if ret:
    cv2.imwrite('all.jpg', frame)
    print("图片已保存为 all.jpg")
else:
    print("无法捕获图像")

cap.release()

