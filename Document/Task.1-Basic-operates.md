**Task.1-Basic-operates**
这是我在实验室上交的第一次作业，本次作业让我窥见一点“视觉”和“电脑”之间的关系。

## 代码拍照
当运用代码调用电脑摄像头拍摄照片在我电脑上成功运行时我感到很惊愕，看到任务里“使用电脑摄像头拍摄手机”时我名字为“all.jpg"的第一张图片是用电脑自带的相机实现的。ps图片路径的不同会导致代码正确但报错
![all jpg](https://github.com/user-attachments/assets/6b37e391-3629-4ef6-9cd4-7764858a2ef0)

## 裁剪
嗯...所以我的第一次裁剪用的也是截图（汗）
![phone](https://github.com/user-attachments/assets/dd3f0196-1a68-4420-a4a9-9f82e223058a)

## 缩放
因为我的all是横屏拍照，但是手机是竖着的，当时很疑惑怎么才能在竖屏的情况下保持完整的手机并符合大小呢。也学到了图像输出像素数据的x,y。
![phone_resized](https://github.com/user-attachments/assets/77668c2e-5cdc-46a5-870d-f25dc3b957ca)

##颜色变换
###彩色模型
数字图像处理中常用的采用模型是RGB（红，绿，蓝）模型和HSV（色调，饱和度，亮度），RGB广泛应用于彩色监视器和彩色视频摄像机，我们平时的图片一般都是RGB模型。而HSV模型更符合人描述和解释颜色的方式，HSV的彩色描述对人来说是自然且非常直观的。
###OpenCV 图像灰度化是一个常见的图像处理操作，它可以将彩色图像转换为灰度图像。
![gray](https://github.com/user-attachments/assets/66138376-7bd6-446b-a040-520f1b86401d)

###hsv
色调（H：hue）用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°。它们的补色是：黄色为60°，青色为180°,品红为300°；
饱和度（S：saturation）：取值范围为0.0～1.0，值越大，颜色越饱和。
亮度（V：value）：取值范围为0(黑色)～255(白色)。
![blue md](https://github.com/user-attachments/assets/5da0d35c-99c2-4132-8cbe-70135fcb6f22)
![green](https://github.com/user-attachments/assets/89585b2d-aa9c-4d53-9f39-aa1bbeb6cb4f)
![red](https://github.com/user-attachments/assets/a5321d61-7bd3-4ef8-ada7-a8b3e013b110)

##物料
用all的代码给物料拍照
![wujian](https://github.com/user-attachments/assets/0d908c4c-96c6-478a-913f-6937b5a7235e)
相反色的几何中心
![几何中心](https://github.com/user-attachments/assets/5dd28986-38ca-4655-adb0-7a075eabe7c2)

综上，经过第一次任务的学习知道了除电子设备出厂设置外的另一种拍照方式，了解了一点opencv基础的图像处理。
