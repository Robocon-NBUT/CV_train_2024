import cv2
img=cv2.imread('all.jpg')

crop=img[53:269,18:121]
cv2.imshow("crop",crop)
cv2.waitKey()

cv2.imwrite('phone.jpg',img)