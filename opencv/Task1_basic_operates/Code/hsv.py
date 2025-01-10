import cv2
img=cv2.imread("phone_resized.jpg")
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


cv2.imwrite("all_hsv.jpg",hsv)
newimg=cv2.imread("all_hsv.jpg")
cv2.imshow("hsv",newimg)
cv2.waitKey(0)