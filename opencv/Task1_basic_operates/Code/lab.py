import cv2
img=cv2.imread("phone_resized.jpg")
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

cv2.imwrite("all_lab.jpg",lab)
newimg=cv2.imread("all_lab.jpg")
cv2.imshow("lab",newimg)
cv2.waitKey(0)