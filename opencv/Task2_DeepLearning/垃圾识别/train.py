# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'C:\Users\许奇峰\PycharmProjects\PythonProject\ultralytics-8.3.59\ultralytics\cfg\models\11\yolo11.yaml')
    model.train(data=r'C:\Users\许奇峰\PycharmProjects\PythonProject\data.yaml',
                imgsz=640,
                epochs=100,
                batch=6,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project=r'C:\Users\许奇峰\PycharmProjects\PythonProject\runs\train',
                name='exp',
                single_cls=False,
                cache=False,
                )
