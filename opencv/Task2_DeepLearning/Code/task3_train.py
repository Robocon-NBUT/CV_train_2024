import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO(model=r'E:\download\ultralytics-8.3.59\ultralytics-8.3.59\ultralytics\cfg\models\11\yolo11.yaml')
    model.train(data=r'data.yaml',
                imgsz=640,
                epochs=300,
                batch=64,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )
