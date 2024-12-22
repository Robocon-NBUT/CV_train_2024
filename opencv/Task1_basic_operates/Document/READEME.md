# Task 1 Basic operates

## 学习内容

> 图像输入操作
> 
> 使用cv2.imread("")文件名

> 图像输出操作
> 
> 使用cv2.imwrite("")函数

> 图像尺寸变换
> 
> 使用cv2.resize(图片名,(尺寸))

> 色彩空间变换
> 
> 使用cv2.cvtColor(image, 色彩空间名,插值方式)

> 颜色识别
> 
> mask = cv2.inRange(hsv_image, lower_color, upper_color)
> 
> contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
> 
> 创建掩膜，随后找到颜色区域

> 二值化，图像平滑算法，形态学操作
> 
> 二值化：参数：原图像，阈值，最大值，二值化类型
ret, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
> 
> 图像平滑算法：均值，中值，高斯，双边滤波等算法
> 
> 形态学：开闭运算，顶黑帽以及结构化元素，可以用于消除噪点等操作

> 边缘检测
> 
> Sobel边缘检测算法是一种用于边缘检测的离散微分算子。它通过计算图像灰度函数的近似梯度来工作。在图像的任何一点使用此算子，将会产生对应的梯度矢量或是其法矢量。Sobel算子结合了高斯平滑和微分求导，用来计算图像明暗程度的近似值，根据图像边缘旁边明暗程度把该区域内超过某个数的特定点记为边缘。
> 
> sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5) # X方向Sobel
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5) # Y方向Sobel
> 
> Canny边缘检测算法是一种多级边缘检测算法，被许多人认为是边缘检测中最优的算法之一。Canny算法的目标是找到一个最优的边缘检测算法，即低错误率、高定位性和最小响应。Canny算法包括滤波、增强和检测三个步骤，通常使用高斯滤波器来消除噪声，然后计算梯度幅值和方向，最后通过阈值化方法来检测和连接边缘。
> 
> edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

> 轮廓识别
> 
> contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
> 
>  绘制轮廓
> 
> cv.drawContours(img, contours, -1, (0, 255, 0), 3)

## 心得

- python库的opencv相对来说简单一点，但是需要记忆函数和用法比较困难，在运用一个新的东西前需要进行形态学变换，以及消去噪点。
