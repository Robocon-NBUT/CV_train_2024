import warnings
 
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO(r'D:/pytorch/deep learning/yolo11s.pt')
    model.train(
                data=r'D:/pytorch/deep learning/yolo/my_datasets/train1.yaml',
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
                project='D:/pytorch/deep learning',
                name='exp',
                )