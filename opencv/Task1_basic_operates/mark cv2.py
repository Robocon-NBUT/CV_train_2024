import cv2
import numpy as np

def get_opposite_color(color):
    """
    获取颜色的相反颜色（BGR 格式）。
    """
    return (255 - color[0], 255 - color[1], 255 - color[2])

# 加载默认物料图像（物料模板）
wuliao2_template = cv2.imread("wuliao2.jpg")
if wuliao2_template is None:
    print("无法读取 wuliao2.jpg 文件")
    exit()

# 转换为 Lab 颜色空间
wuliao2_lab = cv2.cvtColor(wuliao2_template, cv2.COLOR_BGR2Lab)

# 打开摄像头
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("无法打开摄像头")
    exit()

print("按 'q' 退出程序，匹配到物料时会保存为 wuliao2_mark.jpg")

while True:
    # 读取摄像头帧
    ret, frame = camera.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    # 转换摄像头帧为 Lab 颜色空间
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

    # 模板匹配
    result = cv2.matchTemplate(frame_lab[:, :, 1], wuliao2_lab[:, :, 1], cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # 如果匹配值大于阈值，则认为检测到目标
    threshold = 0.7  # 可以根据实际效果调整阈值
    if max_val > threshold:
        # 获取匹配区域的坐标和大小
        h, w, _ = wuliao2_lab.shape
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # 外接矩形的中心
        center_x = (top_left[0] + bottom_right[0]) // 2
        center_y = (top_left[1] + bottom_right[1]) // 2

        # 随机生成轮廓颜色
        color = tuple(np.random.randint(0, 255, 3).tolist())
        opposite_color = get_opposite_color(color)

        # 绘制物料外接矩形轮廓
        cv2.rectangle(frame, top_left, bottom_right, opposite_color, 2)

        # 绘制几何中心的十字标记
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), color, 2)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), color, 2)

        # 保存标记后的图片
        cv2.imwrite("wuliao_mark.jpg", frame)
        print("已保存标记后的图片为 wuliao_mark.jpg")

    # 显示结果
    cv2.imshow("Camera", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
camera.release()
cv2.destroyAllWindows()
