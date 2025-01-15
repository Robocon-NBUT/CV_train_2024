import cv2
import numpy as np
image_path = "D:/PycharmProjects/opencv/all.jpg"  #图片路径
img=cv2.imread(image_path)
if img is None:
  print ("图像加载失败")
else:
    print("图像加载成功")
roi = cv2.selectROI("选择手机区域", img, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("选择手机区域")
x, y, w, h = roi
if w == 0 or h == 0:
    print("未选择任何区域。")
    exit()
phone_image = img[y:y+h, x:x+w]
output_path = 'phone.jpg'
cv2.imwrite(output_path, phone_image)
cv2.imshow("截取的手机图像",phone_image)
cv2.waitKey(0)


