import cv2

image_path = "D:/PycharmProjects/opencv/all.jpg"  #图片路径
img=cv2.imread(image_path)
if img is None:
  print ("图像加载失败")
else:
    print("图像加载成功")
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv_image=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab_image=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

cv2.imwrite('all_gray.jpg', gray_image)
cv2.imwrite('all_hsv.jpg', hsv_image)
cv2.imwrite('all_lab.jpg', lab_image)
cv2.waitKey(0)