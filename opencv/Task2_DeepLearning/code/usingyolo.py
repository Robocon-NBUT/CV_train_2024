from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO(r"C:\Users\柳PC\Desktop\markdown\runs\train\exp4\weights\best.pt") 

if model is None:
    print("模型加载失败，请检查模型路径。")
    exit()
    
# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 读取摄像头的帧
    ret, frame = cap.read()
    if ret:
        # 对帧进行预测
        results = model(frame, save=False) 

        # 遍历检测结果
        for result in results:
            annotated_frame = result.plot()

        # 显示图像
        cv2.imshow('YOLO Detect Result', annotated_frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 释放摄像头资源
cap.release()
# 关闭所有窗口
cv2.destroyAllWindows()