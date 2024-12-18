import cv2

all_img = cv2.imread('all.jpg')

gray_img = cv2.cvtColor(all_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('all_gray.jpg', gray_img)

hsv_img = cv2.cvtColor(all_img, cv2.COLOR_BGR2HSV)
cv2.imwrite('all_hsv.jpg', hsv_img)

lab_img = cv2.cvtColor(all_img, cv2.COLOR_BGR2Lab)
cv2.imwrite('all_lab.jpg', lab_img)
