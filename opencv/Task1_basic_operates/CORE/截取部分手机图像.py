import cv2

# 需要裁剪的图片路径
infile = 'all.jpg'
# 裁剪后图片的保存路径
outfile = 'phone.jpg'

# 原始图像
img = cv2.imread(infile)

# 设定裁剪区域的左上角坐标 (x, y) 以及裁剪后的宽和高 (width, height)
x = 100  # 这里示例左上角x坐标，可按需修改
y = 100  # 这里示例左上角y坐标，可按需修改
width = 300  # 裁剪区域的宽度，可按需修改
height = 200  # 裁剪区域的高度，可按需修改

# 进行裁剪
img_cropped = img[y:y + height, x:x + width]

# 保存裁剪后的图片
cv2.imwrite(outfile, img_cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()
