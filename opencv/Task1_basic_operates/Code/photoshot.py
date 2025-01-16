import cv2
import time
import os
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break
    cv2.imshow('Camera Feed',frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        print("即将进行截图，请保持画面稳定...")
        time.sleep(3)  # 等待3秒，这里的时间可以根据需要自行调整
        cv2.imwrite('screenshot.jpg', frame)
        save_path = os.path.join(desktop_path, "photo.jpg")
        cv2.imwrite(save_path, frame)
        print("截图已成功保存")
        break