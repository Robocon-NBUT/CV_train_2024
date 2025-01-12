import cv2
import numpy as np

# 打开摄像头并进行错误处理
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头，请检查设备连接或权限设置。")
    exit(1)

# 读取一帧并进行错误处理
ret, frame = cap.read()
if not ret:
    print("读取摄像头帧失败，请重试。")
    cap.release()
    exit(1)

# 保存图片，若保存失败进行错误处理
success = cv2.imwrite('2.jpg', frame)
if not success:
    print("保存图像文件失败，请检查磁盘空间或文件权限。")
    cap.release()
    exit(1)

# 释放摄像头
cap.release()

# 读取图像文件，若读取失败进行错误处理
image = cv2.imread('2.jpg')
if image is None:
    print("无法读取保存的图像文件，请检查文件是否存在及路径是否正确。")
    exit(1)

# 转换为灰度图像（假设物料与背景有明显的灰度差异）
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 自适应阈值处理，获取物料的二值图像（改进阈值确定方式，这里使用自适应阈值，可根据图像局部情况确定阈值）
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# 查找轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 定义绘制轮廓的颜色（这里选择绿色，可根据需要修改）
contour_color = (0, 255, 0)

# 遍历每个轮廓
for contour in contours:
    # 计算外接矩形
    x, y, w, h = cv2.boundingRect(contour)
    # 计算外接矩形的几何中心
    center_x = x + w // 2
    center_y = y + h // 2
    # 用指定颜色绘制轮廓
    cv2.drawContours(image, [contour], -1, contour_color, 2)
    # 用与轮廓相同颜色绘制几何中心的十字
    cv2.line(image, (center_x - 5, center_y), (center_x + 5, center_y), contour_color, 2)
    cv2.line(image, (center_x, center_y - 5), (center_x, center_y + 5), contour_color, 2)

# 显示结果
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果图像为3.jpg
cv2.imwrite('3.jpg', image)

# 释放相关内存（删除不再使用的中间变量，优化内存管理）
del frame
del gray
del thresh
del contours