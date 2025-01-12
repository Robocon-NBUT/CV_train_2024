import cv2
import numpy as np

def get_opposite_color(color):
    """
    获取颜色的相反颜色（BGR 格式）。
    """
    return (255 - color[0], 255 - color[1], 255 - color[2])

# 加载默认物料图像（wuliao2模板）
wuliao2_template = cv2.imread("wuliao2.jpg")
if wuliao2_template is None:
    print("无法读取 wuliao2.jpg 文件")
    exit()

# 将模板图像转换为 HSV 颜色空间
wuliao2_hsv = cv2.cvtColor(wuliao2_template, cv2.COLOR_BGR2HSV)

# 打开摄像头
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("无法打开摄像头")
    exit()

print("按 'q' 退出程序，匹配到物料时会保存为 wuliao_mark.jpg")

while True:
    # 读取摄像头帧
    ret, frame = camera.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    # 将摄像头帧转换为 HSV 颜色空间
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 模板匹配
    result = cv2.matchTemplate(frame_hsv, wuliao2_hsv, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # 如果匹配值大于阈值，则认为检测到目标
    threshold = 0.36  # 建议阈值调整
    if max_val > threshold:
        print(f"匹配成功，匹配值：{max_val:.2f}")  # 调试输出匹配值

        # 获取匹配区域的坐标和大小
        h, w, _ = wuliao2_hsv.shape
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # 绘制矩形框
        opposite_color = get_opposite_color((0, 255, 0))
        cv2.rectangle(frame, top_left, bottom_right, opposite_color, 2)

        # 外接矩形的中心
        center_x = (top_left[0] + bottom_right[0]) // 2
        center_y = (top_left[1] + bottom_right[1]) // 2

        # 绘制几何中心的十字标记
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), opposite_color, 2)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), opposite_color, 2)

        # 提取匹配区域并在匹配区域内寻找轮廓
        match_region = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        match_region_gray = cv2.cvtColor(match_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(match_region_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 查找匹配区域的轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制轮廓
        for contour in contours:
            # 将局部轮廓坐标转换为全局坐标
            contour += np.array([top_left[0], top_left[1]])
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

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
