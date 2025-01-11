import cv2
img=cv2.imread("task1\kitty.jpg")
#res=cv2.resize(img,(132,150))
res=cv2.resize(img,None,fy=0.5,fx=0.5,interpolation=cv2.INTER_LINEAR)
#cv2.imshow("hello",img),cv2.imshow("hellokitty",res)
#cv2.waitKey(0)
import numpy as np
dst=cv2.flip(res,1)

cv2.imshow("flipkitty",np.hstack((res,dst)))
cv2.waitKey(0)