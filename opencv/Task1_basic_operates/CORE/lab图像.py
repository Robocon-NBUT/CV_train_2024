import numpy as np
import cv2

# 读取图片
pic_file = 'all.jpg'
img_bgr = cv2.imread(pic_file, cv2.IMREAD_COLOR)

# 将BGR图片转换为LAB图片
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

# 保存LAB色彩空间的图片
cv2.imwrite("all_lab.jpg", img_lab)

# 创建窗口并显示LAB色彩空间的图片
cv2.namedWindow("input", cv2.WINDOW_GUI_NORMAL)
cv2.imshow("input", img_lab)

cv2.waitKey(0)
cv2.destroyAllWindows()