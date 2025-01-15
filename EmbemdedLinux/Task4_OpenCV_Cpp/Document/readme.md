# 学习心得
## cppp调用opencv
### windows与ububtu的文件传输
docker cp [选项] 容器名称或容器 ID:源文件或源目录的路径 目标路径
docker cp [选项] 源文件或源目录的路径 容器名称或容器 ID:目标路径
### cpp截取手机
#### cv::imread
等同于cv2.imread，用来读取图片
#### cv::Rect
cv::Rect是OpenCV中用于表示矩形区域的类。它可以通过指定矩形左上角的坐标和矩形的宽度和高度来创建一个矩形区域。用来框定截取范围
#### cv::imwrite
等同于cv2.imwrite，用来保存图片
### 尺寸缩放
#### cv::Point2f
cv::Point2f和cv::getRotationMatrix2D是OpenCV中用于表示二维点的类。它可以通过指定点的x和y坐标来创建一个二维点。等同cv2.getRotationMatrix2D
#### cv::warpAffine
cv::warpAffine是OpenCV中用于进行仿射变换的函数。它可以对图像进行旋转、缩放、平移等操作。
### 颜色空间的转换
#### cv::cvtColor
cv::cvtColor是OpenCV中用于颜色空间转换的函数。它可以将图像从一种颜色空间转换为另一种颜色空间。
### 标注几何中心
#### cv::Scalar 类和 cv::inRange 函数
cv::Scalar 用于表示颜色值。
cv::inRange 用于根据上下界生成二值化掩码，将在上下界范围内的像素设为 255，其余设为 0。
#### cv::medianBlur 函数
cv::medianBlur 用于对图像进行中值滤波，去除图像中的椒盐噪声。
#### cv::GaussianBlur 函数
cv::GaussianBlur 用于对图像进行高斯滤波，去除图像中的高斯噪声。
#### cv::bitwise_and 函数
cv::bitwise_and 用于对图像进行按位与操作，将两个图像的像素值进行与运算。
#### cv::Canny
cv::Canny 用于进行边缘检测。
#### cv::findContours 函数
cv::findContours 用于查找图像中的轮廓。
#### cv::contourArea
cv::contourArea 用于计算轮廓的面积。
#### cv::boundingRect 
cv::boundingRect 用于计算轮廓的外接矩形。
#### cv::mean 
cv::mean 用于计算图像的均值。
#### cv::drawContours
cv::drawContours 用于绘制轮廓。
#### cv::minAreaRect
cv::minAreaRect 用于计算最小外接矩形。
#### cv::RotatedRect::points
cv::RotatedRect::points 用于获取最小外接矩形的四个顶点。
#### cv::Point 类和 cvRound 函数
cv::Point 用于表示二维点。
cvRound 用于四舍五入。
#### cv::line
cv::line 用于绘制直线。

<img width="429" alt="2a0a9f2363c831df8c68f820d2acc91" src="https://github.com/user-attachments/assets/b060e766-e134-42ea-8e2b-4cd73b9b09fe" />
<img width="758" alt="6a638b9312a47e999265cd1a6bb41a8" src="https://github.com/user-attachments/assets/5188903d-ff53-4b0d-be2f-<img width="753" alt="2c105106ddf0a1bab0aa4541cee5f69" src="https://github.com/user-attachments/assets/e24151af-5c8b-467a-9d47-9741afe2e2a1" />
00bb31a61558" />
<img width="752" alt="e1adf6549e698c2758b3b9798bd8ae0" src="https://github.com/user-attachments/assets/0d95d4e7-8dc6-4f10-808c-c2f3647dd534" />
<img width="840" alt="ecb0eafc9adbfc33943713affd7ad04" src="https://github.com/user-attachments/assets/b8052731-536b-41c8-9fda-8a7a4828e21b" />





