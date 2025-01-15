import cv2
from ultralytics import YOLO


# 加载训练好的模型
model = YOLO(r'C:\Users\35453\Desktop\ultralytics-8.3.2\runs\train\exp14\weights\best.pt')

# 打开电脑摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 逐帧读取摄像头画面
    success, frame = cap.read()
    if success:
        # 使用模型进行预测
        results = model(frame)

        # 可视化预测结果
        annotated_frame = results[0].plot()

        # 显示标注后的画面
        cv2.imshow('Real-time object detection', annotated_frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()