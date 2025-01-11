import cv2
image = cv2.imread('task1/all.jpg')
#cv2.imshow('image', image)
#cv2.waitKey(0)
newimage=image[30:480,200:490]
cv2.imwrite("phone.jpg",newimage)
phone=cv2.imread("phone.jpg")
cv2.imshow("phone",phone)
cv2.waitKey(0)