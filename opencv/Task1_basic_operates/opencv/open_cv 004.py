import cv2
import os

# 获取当前脚本所在的目录
script_directory = os.path.dirname(os.path.abspath(__file__))

# 创建输出文件夹，保存在脚本所在目录中
output_folder = os.path.join(script_directory, 'captured_images')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取 all.jpg
all_path = os.path.join(output_folder, 'all.jpg')
all_image = cv2.imread(all_path)

# 转换颜色空间
all_gray = cv2.cvtColor(all_image, cv2.COLOR_BGR2GRAY)
all_hsv = cv2.cvtColor(all_image, cv2.COLOR_BGR2HSV)
all_lab = cv2.cvtColor(all_image, cv2.COLOR_BGR2LAB)

# 保存
gray_path = os.path.join(output_folder, 'all_gray.jpg')
hsv_path = os.path.join(output_folder, 'all_hsv.jpg')
lab_path = os.path.join(output_folder, 'all_lab.jpg')

cv2.imwrite(gray_path, all_gray)
cv2.imwrite(hsv_path, all_hsv)
cv2.imwrite(lab_path, all_lab)
print(f"颜色空间图像已保存为: {gray_path}, {hsv_path}, {lab_path}")
