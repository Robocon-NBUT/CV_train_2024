from ultralytics import YOLO
import cv2
import numpy as np

# 加载 YOLOv8 模型
model = YOLO('D:/rubbish/yolov8n.pt')  # 直接加载 yolov8n 模型权重文件

# 设置类别标签
labels = ['可回收物_塑料瓶', '可回收物_易拉罐', '可回收物_纸箱',
          '有害垃圾_电池', '有害垃圾_过期药品', '有害垃圾_废旧灯管',
          '厨余垃圾_青菜', '厨余垃圾_果皮', '厨余垃圾_骨头']

# 启动摄像头
cap = cv2.VideoCapture(0)  # 0 是默认摄像头

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

confidence_threshold = 0.3  # 降低置信度阈值
nms_threshold = 0.5  # 调整NMS阈值

while True:
    # 捕获每一帧
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # 转换图像为 RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 使用 YOLOv8 模型进行推理
    results = model(image_rgb)  # 传入 RGB 图像

    # 获取推理结果
    predictions = results[0].boxes  # 获取第一张图像的预测结果

    # 筛选出置信度高于 0.3 的结果
    boxes = []
    confidences = []
    class_ids = []
    for pred in predictions:
        if pred.conf[0] > confidence_threshold:  # 置信度阈值
            x1, y1, x2, y2 = map(int, pred.xyxy[0])  # 获取目标框的坐标
            boxes.append([x1, y1, x2, y2])
            confidences.append(pred.conf[0].item())  # 置信度
            class_ids.append(int(pred.cls[0]))  # 类别 ID

    if len(boxes) == 0:  # 如果没有识别到任何物体，继续下一帧
        print("No objects detected. Continuing to next frame...")
    else:
        # 非最大抑制（NMS）去除冗余框
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=confidence_threshold, nms_threshold=nms_threshold)

        # 输出识别的物体信息
        for i in indices.flatten():
            if class_ids[i] < len(labels):  # 检查类别是否在 labels 范围内
                x1, y1, x2, y2 = boxes[i]
                label = labels[class_ids[i]]
                confidence = confidences[i]
                print(f"Detected: {label} with confidence {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")

                # 在图片上绘制识别结果
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                print(f"Warning: Class ID {class_ids[i]} is out of range for labels.")

    # 显示识别结果
    cv2.imshow("Real-Time Object Detection", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
