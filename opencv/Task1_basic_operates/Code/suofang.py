import cv2
img=cv2.imread("phone.jpg")
phone_resized=cv2.resize(img,(480,640))
cv2.imwrite("phone_resized.jpg",phone_resized)
newimg=cv2.imread("phone_resized.jpg")
cv2.imshow("new",newimg)
cv2.waitKey(0)