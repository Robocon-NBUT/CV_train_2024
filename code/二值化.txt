
import cv2

img = cv2.imread('wujian.jpg')
gray=cv2.imread('wujian.jpg',cv2.IMREAD_GRAYSCALE)
ret,binary=cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
binary_adaptive=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)

cv2.imshow("gray",gray)
cv2.imshow("binary",binary)
cv2.imshow("aptiveary",binary_adaptive)

cv2.waitKey()