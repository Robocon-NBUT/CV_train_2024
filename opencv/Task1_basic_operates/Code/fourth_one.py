import cv2

#灰度
op = cv2.imread('all.jpg')
gray_op = cv2.cvtColor(op, cv2.COLOR_BGR2GRAY)
cv2.imwrite('all_gray.jpg', gray_op)

#hsv
hsv_op = cv2.cvtColor(op, cv2.COLOR_BGR2HSV)
cv2.imwrite('all_hsv.jpg', hsv_op)

#lab
lab_op = cv2.cvtColor(op, cv2.COLOR_BGR2LAB)
cv2.imwrite('all_lab.jpg', lab_op)