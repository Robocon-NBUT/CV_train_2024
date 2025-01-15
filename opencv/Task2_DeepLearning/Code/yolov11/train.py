import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.pt')
    model.train(
                data=r'./coco.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,
                batch=8,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD',
                amp=True,
                project='runs/train',
                name='exp',
                )