import cv2
import os
import numpy as np

# 获取当前脚本所在的目录
script_directory = os.path.dirname(os.path.abspath(__file__))

# 创建输出文件夹，保存在脚本所在目录中
output_folder = os.path.join(script_directory, 'captured_images')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取物料文件
wuliao_path = os.path.join(output_folder, 'wuliao.jpg')
wuliao_image = cv2.imread(wuliao_path)

if wuliao_image is None:
    print(f"无法读取物料文件: {wuliao_path}")
    exit()
else:
    print(f"成功读取物料文件: {wuliao_path}")

# 定义获取相反颜色的函数
def get_opposite_color(color):
    return (255 - color[0], 255 - color[1], 255 - color[2])

# 打开摄像头
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("无法打开摄像头")
    exit()

print("按 'q' 退出程序")

while True:
    # 读取摄像头帧
    ret, frame = camera.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    # 转换为灰度图以便于处理
    wuliao_gray = cv2.cvtColor(wuliao_image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 模板匹配
    result = cv2.matchTemplate(frame_gray, wuliao_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # 匹配成功时
    threshold = 0.4
    if max_val > threshold:
        h, w = wuliao_gray.shape
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # 计算相反颜色
        opposite_color = get_opposite_color((0, 255, 0))

        # 绘制矩形框
        cv2.rectangle(frame, top_left, bottom_right, opposite_color, 2)

        # 计算外接矩形的中心
        center_x = (top_left[0] + bottom_right[0]) // 2
        center_y = (top_left[1] + bottom_right[1]) // 2

        # 绘制几何中心的十字标记
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), opposite_color, 2)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), opposite_color, 2)

        # 提取匹配区域并查找轮廓
        matched_region = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        matched_region_gray = cv2.cvtColor(matched_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(matched_region_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制轮廓
        for contour in contours:
            contour += np.array([top_left[0], top_left[1]])  # 转换为全局坐标
            cv2.drawContours(frame, [contour], -1, opposite_color, 2)

    # 显示结果
    cv2.imshow("Camera", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和窗口资源
camera.release()
cv2.destroyAllWindows()
