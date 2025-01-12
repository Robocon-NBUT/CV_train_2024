import cv2

print(cv2.getVersionString())
img=cv2.imread('all.jpg')
print(img.shape)#图片像素

cv2.imshow("img",img)
cv2.waitKey()