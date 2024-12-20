#
<center><font face ="楷体" size=300>opencv基础</font></center>

<center><font  face="楷体" size=5> 谭尧瑞</font></center>
<font face="楷体" size=5>

# 任务1：使用Python版OpenCV库的基础操作



### 1. 使用电脑摄像头拍摄手机，并保存图片为`all.jpg`（10%）
- 打开电脑摄像头，使用OpenCV库捕获实时图像。
- 将捕获到的图像保存为`all.jpg`文件。
![all](https://github.com/user-attachments/assets/fc9cf38c-659a-4d2f-bd13-dd6cb2d9204b)

### 2. 打开`all.jpg`并截取手机部分，保存图片为`phone.jpg`（10%）
- 使用OpenCV打开`all.jpg`文件。
- 根据需要截取的手机部分的坐标，进行裁剪操作。
- 将裁剪得到的图像保存为`phone.jpg`文件。
![phone](https://github.com/user-attachments/assets/ecdb8da0-0c31-406f-9273-b0395d5529dd)

### 3. 打开`phone.jpg`并将尺寸缩放至与`all.jpg`一致，保存图片为`phone_resized.jpg`（20%）
- 打开`phone.jpg`和`all.jpg`文件。
- 获取`all.jpg`的尺寸信息。
- 将`phone.jpg`的尺寸调整至与`all.jpg`一致。
- 保存调整尺寸后的图像为`phone_resized.jpg`文件。
![phone_resized](https://github.com/user-attachments/assets/29ba8930-d767-4217-b117-1c40cffc2ac3)

### 4. 将`all.jpg`保存为灰度、HSV、LAB三种颜色空间各一张，分别命名为`all_gray.jpg`、`all_hsv.jpg`、`all_lab.jpg`（20%）
- 使用OpenCV打开`all.jpg`文件。
- 将图像转换为灰度颜色空间，并保存为`all_gray.jpg`。
- 将图像转换为HSV颜色空间，并保存为`all_hsv.jpg`。
- 将图像转换为LAB颜色空间，并保存为`all_lab.jpg`。
![all_gray](https://github.com/user-attachments/assets/4eb6ec99-4ffd-4dd5-9442-072092ef2b11)
![all_hsv](https://github.com/user-attachments/assets/6bab49d6-18fb-4f7f-a0b3-a84876265519)
![all_lab](https://github.com/user-attachments/assets/042e66cb-a912-46e3-929e-0a174f243a38)

### 5. 在摄像头中用相反的颜色标出给定的物料轮廓，并用与轮廓相同的颜色用十字标出物料外接矩形的几何中心（40%）
- 再次打开电脑摄像头，捕获实时图像。
- 使用颜色识别技术找到给定物料的轮廓。
- 用相反的颜色在图像中标出这些轮廓。
- 计算物料外接矩形的几何中心。
- 在几何中心位置用与轮廓相同的颜色画一个十字标记。


https://github.com/user-attachments/assets/647a823f-171d-4674-adbb-2a9a22fb3eea




## 心得体会
通过这次任务，我对OpenCV库的基本操作有了更深入的了解和实践。学习如何使用摄像头捕获图像、处理图像尺寸和色彩空间的转换，以及如何进行颜色识别和轮廓检测，这些都是图像处理领域非常重要的技能。在实际操作过程中，我遇到了一些挑战，比如如何精确地裁剪图像和调整尺寸，以及如何准确地识别和标记轮廓。这些挑战让我更加明白了理论知识与实践操作之间的差异，也让我学会了如何通过不断尝试和调整来解决问题。总的来说，这是一次非常宝贵的学习经历，它不仅提高了我的技术能力，也增强了我解决问题的能力。
