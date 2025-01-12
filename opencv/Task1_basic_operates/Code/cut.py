import cv2
image = cv2.imread('all.jpg')
#cv2.imshow('all',image)
cut_image=image[200:650,320:600]
cv2.imwrite('phone.jpg',cut_image)
cv2.imshow('phone',cut_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
