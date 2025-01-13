import cv2
from ultralytics import YOLO

# 加载 YOLO 模型（这里需要用你训练好的模型路径）
model = YOLO('yolo11n.pt')

# 分类映射字典，确保映射类别名称
label_mapping = {
    'red_cap': '可回收垃圾 - 红瓶盖',
    'yellow_cap': '可回收垃圾 - 黄瓶盖',
    'orange_cap': '可回收垃圾 - 橙瓶盖',
    'yellow_pen': '其他垃圾 - 黄笔盖',
    'transparent_pen': '其他垃圾 - 透明笔盖',
    'black_pen': '其他垃圾 - 黑笔盖',
    'nanfu_battery': '有毒有害垃圾 - 南孚电池',
    'green_battery': '有毒有害垃圾 - 绿电池',
    'purple_battery': '有毒有害垃圾 - 紫电池'
}

# 打开摄像头（0 为默认摄像头）
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头的一帧
    ret, frame = cap.read()

    # 检查帧是否读取成功
    if not ret:
        print("无法读取摄像头视频流")
        break

    # 使用 YOLO 模型进行对象检测
    results = model(frame)

    # 获取预测结果（bounding boxes）
    for result in results:
        boxes = result.boxes  # 获取所有的预测框
        for box in boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]  # 获取框的坐标
            conf = box.conf[0]  # 获取置信度
            cls_index = int(box.cls[0])  # 获取类别索引
            cls_name = model.names[cls_index]  # 获取类别名称

            # 将类别名称映射为中文标签
            cls_name_chinese = label_mapping.get(cls_name, cls_name)

            # 绘制检测框和标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{cls_name_chinese} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示带有检测结果的帧
    cv2.imshow('YOLO Object Detection', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
