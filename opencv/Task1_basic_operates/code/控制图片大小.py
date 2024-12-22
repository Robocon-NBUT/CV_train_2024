import cv2

image_path2 = "D:/PycharmProjects/opencv/phone.jpg"
image_path1 = "D:/PycharmProjects/opencv/all.jpg"
img1=cv2.imread(image_path1)
img2=cv2.imread(image_path2)
if img1 is None:
  print ("图像1加载失败")
else:
    print("图像1加载成功")
if img2 is None:
  print ("图像2加载失败")
else:
    print("图像2加载成功")


height1, width1 = img1.shape[:2]
target_size = (width1, height1)


resized_image2 = cv2.resize(img2, target_size, interpolation=cv2.INTER_LINEAR)
cv2.imwrite('phone_resized.jpg',resized_image2 )

