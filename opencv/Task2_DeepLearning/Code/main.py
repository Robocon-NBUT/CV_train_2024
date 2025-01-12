import warnings 
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

if __name__=='__main__':
    model=YOLO('yolo11s.pt')
    early_stop=EarlyStopping(
        monitor='val/box_loss',
        patience=10,
        mode='min',
        verbose=True
    )
    trainer=pl.Trainer(
        callbacks=[early_stop],
        max_epochs=150
    )
    model.train(data='./datas/detect/czdsb.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                single_cls=False,
                batch=32,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD',
                amp=True,
                project='./datas/detect/runs/train',
                name='num1',
                )
    if early_stop.stopped_epoch > 0:
        print(f"训练提前结束在第{early_stop.stopped_epoch}轮")
    else:
        print("训练正常结束")