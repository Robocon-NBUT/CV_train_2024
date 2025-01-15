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
    model = YOLO(model=r'C:\Users\邓子叶\Desktop\deep_learning\ultralytics-8.3.2\runs\train\exp15\weights\best.pt')
    model.predict(source=0,
                  save=True,
                  show=True,
                  )