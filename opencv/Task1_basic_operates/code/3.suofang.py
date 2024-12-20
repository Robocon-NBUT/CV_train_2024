import cv2





#缩放
img1=cv2.imread('phone.jpg')
img3=cv2.imread('all.jpg')

cv2.imshow('phone',img3)
print (img3.shape)

print(img1.shape)

img2=cv2.resize(img1,(640,480))
cv2.imshow('gai',img2)

cv2.waitKey(0)