import cv2

image = cv2.imread('all.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('all_gray.jpg', gray_image)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imwrite('all_hsv.jpg', hsv_image)

lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imwrite('all_lab.jpg', lab_image)
