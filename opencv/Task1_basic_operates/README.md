# Task 1 Basic operates

## 学习内容

- python版opencv库的安装与使用 ：           
对pyhton 3.13版本的安装和安装pycharm，以及对其的激活
  
- 图像输入操作 ：                                                                                                                                  
此任务点同下一个任务点为一个任务，先导入cv2模块，在用img来代表储存图像变量，在用imread对图片进行读取输入，最后用imshow对图片进行输出
- 图像输出操作
  
- 图像尺寸变换 ：                             
常用cv2.resize函数实现，需要指定要变换尺寸的原始图像、目标尺寸以及特设值。
  
- 色彩空间变换 ：
灰度变换：通过cv2.cvtColor函数，传入原始图像以及指定的色彩空间转换代码（将彩色图像转灰度时常用cv2.COLOR_BGR2GRAY代码），可将彩色图像转换为灰度图像。
HSV、LAB 等变换：同样借助cv2.cvtColor函数，使用对应的色彩空间转换代码（如cv2.COLOR_BGR2HSV、cv2.COLOR_BGR2LAB），
可以实现从 BGR 色彩空间到 HSV、LAB 等其他色彩空间的转换，方便后续依据不同色彩空间特性进行相应的颜色处理与分析。
  
- 颜色识别 ：                                                                                                          
一般基于特定色彩空间下对颜色范围的界定，通过对像素值进行条件判断等方式来识别图像中特定颜色区域，
例如在 HSV 色彩空间中，设定色调、饱和度、明度的范围来确定某种颜色所在区域，常结合阈值处理、掩码操作等来实现精确的颜色识别效果。
  
- 二值化，图像平滑算法，形态学操作 ：                                                                 
二值化：利用cv2.threshold函数，设定合适的阈值和阈值类型（如固定阈值、自适应阈值等），将图像像素值根据阈值划分成黑白两色（0 和 255），使图像呈现二值化效果，便于后续目标提取、特征分析等操作。
图像平滑算法：像均值滤波（cv2.blur函数）、高斯滤波（cv2.GaussianBlur函数）、中值滤波（cv2.medianBlur函数）等，通过对图像像素邻域进行不同方式的运算，去除图像中的噪声，使图像更加平滑，提升图像质量，利于后续准确的特征提取与分析。
形态学操作：基于数学形态学原理，使用如腐蚀（cv2.erode函数）、膨胀（cv2.dilate函数）、开运算（先腐蚀后膨胀）、闭运算（先膨胀后腐蚀）等操作，改变图像中目标的形状、去除小的噪声点或者填补孔洞等，常用于目标检测、图像分割等场景中对目标物体的预处理与优化。
  
- 边缘检测 ：                                                                                                                                 
常见的有cv2.Canny边缘检测算法，通过设定高低阈值等参数，检测图像中像素强度变化剧烈的地方（即边缘），返回边缘图像，能有效提取图像中物体的轮廓边界信息，为后续的轮廓识别、目标定位等提供基础。
  
- 轮廓识别：                                                                                                           
通过cv2.findContours函数查找图像中的轮廓，返回轮廓信息以及轮廓的层次结构等，再结合相关函数可以对轮廓进行绘制、分析其几何特征（如计算外接矩形、几何中心等），还可以利用轮廓相关的属性进行目标识别、形状匹配等。

![all](https://github.com/user-attachments/assets/7b631f52-8365-4dc8-bc17-24e7ee5c3109)
![phone](https://github.com/user-attachments/assets/24a5859b-8822-48ce-8175-444765ac5348)
![phone_resized](https://github.com/user-attachments/assets/6244455f-01c1-48d2-8418-e1ff9c070d03)
![all_gray](https://github.com/user-attachments/assets/6bb673d2-ee38-4b4f-b15a-fad118b24424)
![all_lab](https://github.com/user-attachments/assets/3001b9d5-5e7d-4fa6-be95-6d4069476831)
![all_hsv](https://github.com/user-attachments/assets/5ba3e1ec-be4c-4464-8e07-4e1d38122490)


https://github.com/user-attachments/assets/7f5736a5-79c9-47d6-9e1d-8d87e92d5e26








