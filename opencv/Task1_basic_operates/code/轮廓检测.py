import cv2

img=cv2.imread("wujian.jpg")
gray=cv2.imread('wujian.jpg',cv2.IMREAD_GRAYSCALE)
ret,thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imshow("thresh",thresh)
cv2.waitKey()
