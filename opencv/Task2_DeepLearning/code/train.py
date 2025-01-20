import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO(r'D:\yolov11\ultralytics-8.3.2\yolo11n.pt')
    model.train(
                data=r'D:\yaml\coco.yaml',  # 修改为正确的绝对路径
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,
                batch=8,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD',
                amp=True,
                project='runs/train',
                name='exp',
                )
