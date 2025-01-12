import cv2
img=cv2.imread('all.jpg')
cv2.imshow("blue",img[:,:,0])
cv2.imshow("green",img[:,:,1])
cv2.imshow("red",img[:,:,2])

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度图
cv2.imshow("gray",gray)

cv2.imshow("img",img)
cv2.waitKey()