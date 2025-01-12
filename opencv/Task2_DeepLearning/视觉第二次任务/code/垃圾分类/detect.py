# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：detect.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
"""

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'D:\Users\黄广松\Desktop\HSGS-yoloV11\runs\train\exp3\weights\best.pt')
    model.predict(source=0,
                  save=True,
                  show=True,
                  )