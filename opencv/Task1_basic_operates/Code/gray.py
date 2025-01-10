import cv2
img=cv2.imread("phone_resized.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite("all_gray.jpg",gray)
newimg=cv2.imread("all_gray.jpg")
cv2.imshow("gray",newimg)
cv2.waitKey(0)