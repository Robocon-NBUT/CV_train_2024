import cv2
e=cv2.imread('all.jpg')
image=cv2.imread('phone.jpg')
re=cv2.resize(image,(e.shape[1],e.shape[0]))
cv2.imwrite('phone_resized.jpg',re)
cv2.imshow('image',re)
cv2.waitKey(0)
cv2.destroyAllWindows()