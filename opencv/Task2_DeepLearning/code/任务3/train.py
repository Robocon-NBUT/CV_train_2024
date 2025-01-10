
import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model=YOLO("yolo11n.yaml").load("my_model\\yolo11n.pt")
    model.train(data=r'data.yaml',
                imgsz=640,
                epochs=200,
                batch=25,
                workers=0,
                device='',
                optimizer='Adam',
                close_mosaic=10,
                resume=False,
                project='D:/1/yolo/runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )

