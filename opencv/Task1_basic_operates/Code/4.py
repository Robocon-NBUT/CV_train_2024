import cv2

# 读取 all.jpg
all_img = cv2.imread('all.jpg')

if all_img is None:
    print("无法加载 all.jpg，请检查文件路径。")
    exit()

# 转换为灰度图
all_gray = cv2.cvtColor(all_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('all_gray.jpg', all_gray)

# 转换为 HSV 图
all_hsv = cv2.cvtColor(all_img, cv2.COLOR_BGR2HSV)
cv2.imwrite('all_hsv.jpg', all_hsv)

# 转换为 LAB 图
all_lab = cv2.cvtColor(all_img, cv2.COLOR_BGR2LAB)
cv2.imwrite('all_lab.jpg', all_lab)

print("已将 all.jpg 保存为灰度图、HSV 图和 LAB 图，分别命名为 all_gray.jpg、all_hsv.jpg 和 all_lab.jpg。")
