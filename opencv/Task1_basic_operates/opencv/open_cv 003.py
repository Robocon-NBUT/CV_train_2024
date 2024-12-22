import cv2
import os

# 获取当前脚本所在的目录
script_directory = os.path.dirname(os.path.abspath(__file__))

# 创建输出文件夹，保存在脚本所在目录中
output_folder = os.path.join(script_directory, 'captured_images')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取 all.jpg 和 wuliao.jpg
all_path = os.path.join(output_folder, 'all.jpg')
wuliao_path = os.path.join(output_folder, 'wuliao.jpg')
all_image = cv2.imread(all_path)
wuliao_image = cv2.imread(wuliao_path)

# 调整尺寸并保存
all_height, all_width = all_image.shape[:2]
wuliao_resized = cv2.resize(wuliao_image, (all_width, all_height))
resized_path = os.path.join(output_folder, 'phone_resized.jpg')
cv2.imwrite(resized_path, wuliao_resized)
print(f"已将 wuliao.jpg 调整尺寸并保存为: {resized_path}")
