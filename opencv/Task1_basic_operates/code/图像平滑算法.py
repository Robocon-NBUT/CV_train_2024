import cv2

image=cv2.imread('wujian.jpg')

gauss=cv2.GaussianBlur(image,(5,5),0)

cv2.imshow("image",image)
cv2.imshow("gauss",gauss)
cv2.waitKey()