# 学习心得
## 一、图像输入操作
使用cv2.imread()函数可以读取本地图像文件。例如：img = cv2.imread('image.jpg')，其中'image.jpg'是图像文件的路径，该函数返回一个表示图像的 NumPy 数组。
还可以从摄像头等视频源获取图像帧，通过cv2.VideoCapture()来实现。例如：cap = cv2.VideoCapture(0)用于打开默认摄像头，然后通过cap.read()方法可以逐帧读取图像。
## 二、图像输出操作
使用cv2.imshow()函数可以在窗口中显示图像。例如：cv2.imshow('Window Name', img)，其中'Window Name'是窗口的标题，img是要显示的图像。
使用cv2.imwrite()函数可以将图像保存到本地。例如：cv2.imwrite('output.jpg', img)可以将img图像保存为output.jpg文件。
## 三、图像尺寸变换
使用cv2.resize()函数可以对图像进行缩放。例如：resized_img = cv2.resize(img, (new_width, new_height))，其中img是原始图像，(new_width, new_height)是目标尺寸。
裁剪图像可以通过对图像数组进行索引操作实现。例如：cropped_img = img[y:y + height, x:x + width]，其中(x, y)是裁剪区域的左上角坐标，width和height是裁剪区域的宽度和高度。
## 四、色彩空间变换
使用cv2.cvtColor()函数进行色彩空间转换。例如，将 RGB 图像转换为灰度图：gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)，将 RGB 图像转换为 HSV 图像：hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)。
## 五、颜色识别
在不同色彩空间（如 HSV、Lab）中设置颜色阈值，然后使用cv2.inRange()函数创建掩码。例如，在 HSV 空间中识别蓝色：
首先设定蓝色在 HSV 中的阈值：lower_blue = np.array([100, 50, 50])，upper_blue = np.array([140, 255, 255])。
然后创建掩码：mask = cv2.inRange(hsv_img, lower_blue, upper_blue)，通过该掩码可以提取出图像中的蓝色区域。
## 六、二值化，图像平滑算法，形态学操作
### 二值化
使用cv2.threshold()函数。例如：ret, binary_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)，其中gray_img是灰度图像，threshold_value是阈值，255是二值化后目标值，cv2.THRESH_BINARY是二值化类型。
### 图像平滑算法
常见的方法有均值滤波cv2.blur()、中值滤波cv2.medianBlur()、高斯滤波cv2.GaussianBlur()等。例如，中值滤波：smoothed_img = cv2.medianBlur(img, kernel_size)，其中kernel_size是滤波核的大小。
### 形态学操作
使用cv2.erode()、cv2.dilate()、cv2.morphologyEx()等函数。例如，对二值图像进行腐蚀操作：eroded_img = cv2.erode(binary_img, kernel)，其中kernel是结构元素。
## 七、边缘检测
常见的边缘检测算子有 Canny 算子。使用cv2.Canny()函数，例如：edges = cv2.Canny(gray_img, low_threshold, high_threshold)，其中gray_img是灰度图像，low_threshold和high_threshold是 Canny 算子的阈值。
## 八、轮廓识别
首先进行边缘检测，然后使用cv2.findContours()函数查找轮廓，例如：contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)，其中edges是边缘图像，cv2.RETR_TREE是轮廓检索模式，cv2.CHAIN_APPROX_SIMPLE是轮廓近似方法。
可以使用cv2.drawContours()函数绘制轮廓，例如：cv2.drawContours(img, contours, - 1, (0, 255, 0), 3)，其中img是原始图像，contours是轮廓列表，-1表示绘制所有轮廓，(0, 255, 0)是轮廓颜色，3是轮廓线的厚度。

## 使用电脑摄像头拍摄手机，并保存图片为 all.jpg
## 打开 all.jpg 并截取手机部分，保存图片为 phone.jpg
## 打开 phone.jpg 并将尺寸缩放至与 all.jpg 一致，保存图片为 phone_resized.jpg 
## 将 all.jpg 保存灰度、hsv、lab 三种颜色空间各一张，分别命名为 all_gray.jpg，all_hsv.jpg，all_lab.jpg 
## 在摄像头中用相反的颜色标出给定物料轮廓，并用与轮廓相同的颜色用十字标出物料外接矩形的几何中心


