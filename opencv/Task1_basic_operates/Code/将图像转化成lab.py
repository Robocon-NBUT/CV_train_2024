import cv2

# 读取图像，此处假设图像文件名为 'test.jpg'，你可替换成实际的图像文件名及路径
image = cv2.imread('all.jpg')

# 将读取的图像从BGR色彩空间转换为LAB色彩空间
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# 以下是可选的操作，用于显示转换后的LAB图像（方便查看效果）
cv2.imshow('all_lab', lab_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 若你想保存转换后的LAB图像，可以使用下面这行代码（保存为名为 'lab_result.jpg' 的图像）
cv2.imwrite('all_lab.jpg', lab_image)