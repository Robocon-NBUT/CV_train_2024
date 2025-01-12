import cv2

op1=cv2.imread('phone.jpg')
op2=cv2.resize(op1,(640,480))
cv2.imshow('phone_resized',op2)
cv2.imwrite('phone_resized.jpg',op2)
cv2.waitKey(0)