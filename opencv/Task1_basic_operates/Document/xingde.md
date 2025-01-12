# Task 1 林明月关于openCV的学习心得
## 概括学习内容
 - python 版 opencv库的安装与使用
 - 图像输入、输出操作
 - 图像尺寸变换
 - 色彩空间变换兼颜色识别
 - 边缘检测与轮廓识别
## 学习心得
### 对OpenCV的了解
- OpenCV是一个基于Apache2.0许可发行的跨平台计算机视觉和机器学习软件库，可以运行在Linux、Windows、Android和Mac OS操作系统上。它轻量级而且高效——由一系列 C 函数和少量 C++ 类构成，同时提供了Python、Ruby、MATLAB等语言的接口，实现了图像处理和计算机视觉方面的很多通用算法。
### 通过openCV实现
- 以Python为载体平台的计算机视觉体验。通过代码调控计算机摄像头进行拍摄，照片处理，物体识别的功能。
### 使用过程的困难
- 对于Python语法的生疏
- 没有很好的掌握摄像头范围

## 任务项（5/5）
### 1.使用电脑摄像头拍摄手机，并保存图片为all.jpg 

![all](https://github.com/user-attachments/assets/5c51dee2-0834-4f63-a220-f818dd220408)
### 2.打开all.jpg并截取手机部分，保存图片为phone.jpg 
![phone](https://github.com/user-attachments/assets/33d7f4ef-10c4-40a9-80a1-c77f3660d2a2)
### 3.打开phone.jpg并将尺寸缩放至与all.jpg一致，保存图片为phone_resized.jpg 
![phone_resized](https://github.com/user-attachments/assets/43f4f930-501f-446f-b458-23edf5b5de6f)
### 4.将all.jpp保存灰度、hsv、lab三种颜色空间各一张，分别命名为all_gray.jpg,all_hsv.jpg,all_lab.jpg 
![all_gray](https://github.com/user-attachments/assets/5e2bf218-f30f-4e36-b142-8637f5053d8a)
![all_hsv](https://github.com/user-attachments/assets/07f1a38d-0d4b-46a9-a899-c9e7d83ffec0)
![all_lab](https://github.com/user-attachments/assets/ca08e5d6-d3e6-4686-8977-56a492e88061)
### 5.在摄像头中用相反的颜色标出给定的物料轮廓，并用与轮廓相同的颜色用十字标出物料外接矩形的几何中心 
![3](https://github.com/user-attachments/assets/4ac521b3-bcc1-4e94-a24d-7caf1f45506f)
