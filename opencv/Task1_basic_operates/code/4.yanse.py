import cv2

#将all.jpp保存灰度、hsv、lab三种颜色空间各一张，分别命名为all_gray.jpg,all_hsv.jpg,all_lab.jpg

#灰度
img = cv2.imread('all.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('all_gray.jpg', gray_img)

#hsv
img1 = cv2.imread('all.jpg')
hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
cv2.imwrite('all_hsv.jpg', hsv_img)


#lab
img2 = cv2.imread('all.jpg')
lab_img = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
cv2.imwrite('all_lab.jpg', lab_img)


