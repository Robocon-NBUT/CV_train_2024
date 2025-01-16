# OpenCV C++ 黄世昌学习心得

## 主要函数总结

### 1. 摄像头操作
- **`cv::VideoCapture cap(0);`**  
  打开摄像头，`0` 表示默认摄像头。
- **`cap >> frame;`**  
  从摄像头捕获一帧图像。
- **`cv::imwrite("all.jpg", frame);`**  
  将图像保存为文件。

### 2. 图像裁剪与缩放
- **`cv::Rect roi(x, y, width, height);`**  
  定义感兴趣区域（ROI）。
- **`cv::Mat phone = image(roi);`**  
  裁剪图像。
- **`cv::resize(phone, resized_phone, target_size);`**  
  缩放图像到目标尺寸。

### 3. 颜色空间转换
- **`cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);`**  
  将图像转换为灰度图。
- **`cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);`**  
  将图像转换为 HSV 颜色空间。
- **`cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);`**  
  将图像转换为 Lab 颜色空间。

### 4. 物料轮廓检测与标注
- **`cv::inRange(hsv, lower_color, upper_color, mask);`**  
  根据颜色范围生成掩码。
- **`cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);`**  
  查找轮廓。
- **`cv::boundingRect(contour);`**  
  获取轮廓的外接矩形。
- **`cv::drawContours(image, contours, -1, color, thickness);`**  
  绘制轮廓。
- **`cv::drawMarker(image, center, color, markerType, size, thickness);`**  
  在图像上绘制标记（如十字）。

![all](https://github.com/user-attachments/assets/ec213c28-1c23-4977-b2f9-8df2f74c2fc6)
![phone](https://github.com/user-attachments/assets/4241c211-835e-4e3e-b264-68136b45afea)
![phone_resized](https://github.com/user-attachments/assets/908d5455-d590-443d-b0ce-6cd3d0e9a9fa)
![all_gray](https://github.com/user-attachments/assets/e7ccab1d-ef2c-4105-a6c6-a5d3baf8cee4)
![all_hsv](https://github.com/user-attachments/assets/716609e6-1d69-4482-b5b9-f792cfa5037b)
![all_lab](https://github.com/user-attachments/assets/86df7a7d-6378-4694-b8d1-34259ca45483)
<img width="475" alt="十字标定" src="https://github.com/user-attachments/assets/0e5a8399-fb21-41e7-9b73-1629d2819a40" />



