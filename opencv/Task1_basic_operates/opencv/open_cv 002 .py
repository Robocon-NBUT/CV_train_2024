import cv2
import os

# 获取当前脚本所在的目录
script_directory = os.path.dirname(os.path.abspath(__file__))

# 创建输出文件夹，保存在脚本所在目录中
output_folder = os.path.join(script_directory, 'captured_images')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取图片 all.jpg
all_path = os.path.join(output_folder, 'all.jpg')
frame = cv2.imread(all_path)

# 选择感兴趣区域
roi = cv2.selectROI("Select Region of Interest", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Region of Interest")  # 关闭 ROI 窗口

# 检查 ROI 是否有效并保存
if roi != (0, 0, 0, 0):
    wuliao_image = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    wuliao_path = os.path.join(output_folder, 'wuliao.jpg')
    cv2.imwrite(wuliao_path, wuliao_image)
    print(f"选定区域已保存为: {wuliao_path}")
else:
    print("未选择有效区域！")
