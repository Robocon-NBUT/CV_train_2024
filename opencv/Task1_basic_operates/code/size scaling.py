import cv2
import numpy as np

img = cv2.imread("D:/opencv_test/phone.jpg")



#旋转
m2=cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),90,1)
The_flipped_image=cv2.warpAffine(img,m2,(img.shape[1],img.shape[0]))

# 定义目标尺寸
target_width = 640
target_height = 480

resized_img = cv2.resize(The_flipped_image, (target_width, target_height))
cv2.imshow('resized_img',resized_img)
cv2.waitKey(0)
cv2.imwrite("phone_resized.jpg",resized_img )
