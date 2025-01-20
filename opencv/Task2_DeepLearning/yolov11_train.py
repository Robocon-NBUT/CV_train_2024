from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')  # 从YAML建立一个新模型
    model.load('yolo11n.pt')
    # # 训练模型
    results = model.train(data='data.yaml',
                      epochs=1000, imgsz=640, device='', optimizer='SGD', workers=8, batch=32, amp=True)