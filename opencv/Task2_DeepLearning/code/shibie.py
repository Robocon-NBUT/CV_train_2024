from ultralytics import YOLO
import cv2


def main():
    # 加载训练好的模型
    model = YOLO(r'C:\Users\谭尧瑞\Desktop\ultralytics\runs\train\exp\weights\best.pt')  

    # 打开摄像头，0 表示默认摄像头，可根据需要修改
    cap = cv2.VideoCapture(1)  

    while cap.isOpened():
        # 读取摄像头的一帧图像
        success, frame = cap.read()
        if success:
            # 对当前帧进行预测
            results = model(frame)

            # 在图像上绘制预测结果
            annotated_frame = results[0].plot()

            # 显示带有预测结果的图像
            cv2.imshow("YOLOv11 Detection", annotated_frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()