import cv2

all_img = cv2.imread('all.jpg')
phone_img = cv2.imread('phone.jpg')

height, width = all_img.shape[:2]

resized_phone_img = cv2.resize(phone_img, (width, height))

cv2.imwrite('phone_resized.jpg', resized_phone_img)
