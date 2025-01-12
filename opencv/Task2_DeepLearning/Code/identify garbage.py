import cv2
from ultralytics import YOLO


def detect_garbage(frame, model):
    results = model(frame)
    detected_objects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0].item()
            if conf > 0.5:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0].item())
                label = model.names[cls]
                detected_objects.append(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label}: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame, detected_objects


if __name__ == '__main__':
    # 加载训练好的模型，路径修改为训练后模型权重文件路径
    model = YOLO('./runs/train/exp16/weights/last.pt')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video device")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result_frame, detected_objects = detect_garbage(frame, model)
        print("Detected objects:", detected_objects)
        cv2.imshow('Garbage Detection', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
